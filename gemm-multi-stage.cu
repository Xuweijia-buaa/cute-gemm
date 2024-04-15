#include <cublas_v2.h>
#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>

#include <cute/tensor.hpp>

#include "detail/cublaslt-gemm.h"
#include "detail/data.h"

template <typename Config>
__global__ void /* __launch_bounds__(128, 1) */
gemm_multi_stage(void *Dptr, const void *Aptr, const void *Bptr, int m, int n,
                 int k) {
  using namespace cute;
  using X = Underscore;

  using T = typename Config::T;
  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using SmemLayoutC = typename Config::SmemLayoutC;
  using TiledMMA = typename Config::MMA;

  using S2RCopyAtomA = typename Config::S2RCopyAtomA;
  using S2RCopyAtomB = typename Config::S2RCopyAtomB;
  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;
  using R2SCopyAtomC = typename Config::R2SCopyAtomC;
  using S2GCopyAtomC = typename Config::S2GCopyAtomC;
  using S2GCopyC = typename Config::S2GCopyC;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;

  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;
  int ix = blockIdx.x;
  int iy = blockIdx.y;

  // use Tensor notation to represent device pointer + dimension
  // g上的逻辑矩阵
  Tensor A = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k), // 行主序
                         make_stride(k, Int<1>{}));  // (M, K)
  Tensor B = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k), // 行主序（k stride=1）. 但从(K,N)看，列主序
                         make_stride(k, Int<1>{}));  // (N, K)
  Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(m, n), // 行主序
                         make_stride(n, Int<1>{}));  // (M, N)

  // slice the tensor to small one which is used for current thread block.
  // g全部mem,按照Tile分块
  // 本cudablock, 只选取该cudablock对应的Tile大小 (AB含k个Tile)
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), // 对应的A Tile,是CTile行对应的多个K： (TileM,TileK,k)
                         make_coord(iy, _));  // (kTileM, kTileK, k)
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), // 对应的B Tile,是CTile行对应的多个K： (TileN,TileK,k)，和A中k对应相乘
                         make_coord(ix, _));  // (kTileN, kTileK, k)
  Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}),  // 本cudablock要完整处理的C Tile
                         make_coord(iy, ix));  // (kTileM, kTileN)

  // shared memory
  // 每个cudablock, 对应Tile* stage个共享内存块
  // 每个Tile块，对应K轴迭代拿入的每个TileA,TileB,
  //    每次从global_mem中的K轴读取 (通过G->S)                                
  //    用于读到寄存器中，进行一次TileA*TileB->TileC的新tileMMA计算
  //以前K次迭代，现在用流水线，可以异步执行: 
  //    某一Tile S->R后，就可以发设下一Tile的G->S
  //    初始发射nsage-1个G->S，占用其中nsage-1个share_mem
  //    每个G-S复制完，开始使用share_mem时，就可以发射下一个G->S，使用此时空缺的share_mem.此时nstage个share_mem都 in use
  //    后续又一个G-S复制完，开始使用share_mem时，说明上个S->R用完了。可以再使用空闲的share_mem,加载下一个Tile
  // swizzle
  auto sA = make_tensor(make_smem_ptr(Ashm),
                        SmemLayoutA{});  // (kTileM, kTileK, kStage)  每个stage用一个TileA。 不同stage,对应不同K,可能异步复制
                                         // share_mem layout,来自于swizzle后的结果
  auto sB = make_tensor(make_smem_ptr(Bshm),
                        SmemLayoutB{});  // (kTileN, kTileK, kStage)

  // dispatch TileA/TileB/TileC mma tensor into thread fragment via partition
  // method
  TiledMMA tiled_mma;
  // 本cudablock,处理C Tile
  // K个迭代中，每次TileA,B和C的计算，需要本线程提供的数据
  // 以A为例，每次需要计算A_tile(128,32),而原来tiledmma可以计算(32,16)，因此还需要继续在M,K方向拓展(4,2),形成新的tiledMMA
  auto thr_mma = tiled_mma.get_slice(idx);
  // 本线程需要提供的数据(对于K轴一次TileAB->C的计算): 以每次mma_atom需要提供的数据为基础，在M,K拓展。（寄存器上，目前只是layout,还未复制）
  // 首维对应一次mma_atom，本线程对应的数据(逻辑offset，对应G中复制进来的的某次share上的Tile，某次mma_atom（对应到该线程后的)）
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)  
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
  auto tCrD = thr_mma.partition_fragment_C(gD);           // (MMA, MMA_M, MMA_N)  本线程：（单次mma_atom要写入寄存器的数据，和MK维拓展，以完成一次tileAB-C的计算）

  // fill zero for accumulator
  clear(tCrD);

  // gmem -cp.async-> shm -ldmatrix-> reg

  // S->R的layout
  // S2RCopyAtomA: 32个线程，读取S中逻辑上(16,16)(8.32)的块，到寄存器  （不一定是逻辑16*16?，只是4个88矩阵=256个元素，和G中读取的8*32逻辑矩阵对应）
  //    对应ldmatrix*.x4指令。
  //    32个线程，每个线程读取share_mem中，一个连续的8元素。
  //    其中每8个线程读取的64个元素，对应原始逻辑矩阵中的88块。存在32个线程的不同寄存器中，每个寄存器2个元素。
  //    32个线程，读取的4个88块， 对应原始逻辑矩阵中的88块。存在32个线程的不同寄存器中，每个寄存器，放4个88矩阵中，每个2元素
  //    32个线程，读取的这个16*16的逻辑矩阵，对应mma_atom中要提供的A(16,16). 正好供32个线程，进行一次mma_atom计算
  // 这里直接用tiled_mma，在M,K维拓展，
  //    因为S2RCopyAtomA{}，直接对应一个mma_atom所需的数据（32个线程所需的(16,16)大小的数据）
  //    所以可以直接按tilemma对mma_atom的拓展，来拓展S2RCopyAtomA{}
  //    同样M,N维做了*2拓展，N维度做了*2 value拓展
  //    tilemma，一次需要A(32,16)，而A经过该拓展，只在M维拓展线程，同样可以提供A(32,16)
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);  // 与tiledMMA需要的数据,对应的copyLayout
  // 对源数据sA(128,32,kstage)进行划分。
  // 与tilemma类似， (128,32),需要多次tileamma,因此进一步拓展s2r_tiled_copy_a，形成新的拓展，对应上边新的tiledMMA：
  // (本线程一次S2RCopyAtomA需要提供的数据，在M,K维的拓展，以完成完share_mem上的TileA sA的复制，以完成一次TileAB-C的计算)
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx); // 与新tiledMMA需要的数据,对应的copyLayout（只涉及本线程的）
  // 只对一次TIleA进行划分
  // (本线程一次S2RCopyAtomA需要提供的数据，在M,K维的拓展，以完成K轴一次A_tile,B_tile的完整加载,对应的新tilemma计算，*)
  // (对应上边新tilemma,完成K轴一次tileAB->C的计算，需要的MK次mma_atom，每次mmat_atom需要的S上的数据)
  auto tAsA = s2r_thr_copy_a.partition_S(sA);  // (CPY, CPY_M, CPY_K, kStage)  对Sa进行partion，划分成多次mma_atom需要的数据
  // 已有的寄存器上的内存，转为D需要的layout。用于接收S复制过来的数据
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);  // (CPY, CPY_M, CPY_K)    已有的R

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);  // ? (CPY, CPY_M, CPY_K, kStage)
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);  // ? (CPY, CPY_M, CPY_K)

  // G->S的layout
  // G2SCopyA：以cp.asyc作为每个线程的原子指令。并通过拓展，可以使128个线程，一次读取G中(32,32)的A元素块
  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  // 本cudablock中，需要的tileA(128,32,k)
  // 而G2SCopyA，使本cudabock中的128个线程，一次执行可以读取gA中(32,32)的A元素块
  // 因此对tileA进行划分，按照G2SCopyA粒度，沿M,K维拓展，看需要执行几次，可以完成tileA大小的复制
  // (本线程一次cp.aync要复制的数据，沿着M,K维拓展，*)，以完成整个gA的G->S  （只对应某个Tile）
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)
  // 与上边类似，但对应要写入的S的layout
  // sA本身对应的layout, 是swizzle过的
  // TODO:多级流水线
  auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStage)  partition不涉及多个stage,只涉及当2维，当前Tile

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);  // (CPY, CPY_N, CPY_K, k)
  auto tBsB_copy =
      g2s_thr_copy_b.partition_D(sB);  // (CPY, CPY_N, CPY_K, kStage)

  int itile_to_read = 0;
  int ismem_read = 0;    // 记录share_mem(CPY, CPY_M, CPY_K, kStage)中，读到的stage (累计)，用于区分K轴大Tile
  int ismem_write = 0;   // share_mem中，G-》S已经发射写入的stage(累积)，用于区分K轴大Tile

  // submit kStage - 1 tile
  // gmem -> shm
#pragma unroll
  // 1 最开始，先发射[nstage-1]个[G->S]
  //   对应K轴中，前[nstage-1]个[TileA],[TileB]， [G->S]的复制（异步的）
  for (int istage = 0; istage < kStage - 1; ++istage) {
    cute::copy(g2s_tiled_copy_a, 
               tAgA_copy(_, _, _, istage),   // G中K轴共k个Tile中，第istage个Tile （作为本线程要复制的源数据）
               tAsA_copy(_, _, _, istage));  // 复制到该stage对应的share_mem中（且对应经过swizzle后的smemlayout）
    cute::copy(g2s_tiled_copy_b,
               tBgB_copy(_, _, _, istage),
               tBsB_copy(_, _, _, istage));
    cp_async_fence();// 标记发射结束

    ++itile_to_read; // 标记G的K轴中，已经执行了[G->S]的Tile位置
    ++ismem_write;   // share_mem中共有标记n_stage个slot。 标记每个Tile【G-S】, 写入的slot位置 （下一个还没写的位置。对n_stage取余数）
                     // 用完n_stage个slot后，某个slot对应的share_mem，完成了S->R后，其slot可以空出来，供新的Tile使用(G->S)
  }

  // wait one submitted gmem->smem done
  cp_async_wait<kStage - 2>();  // 有一个G->S结束了 (首个Tile)。可以开始该share_mem的 S->R (拆成多个小k,二级流水，按照mma_atom粒度)
  __syncthreads();

  int ik = 0;
  // 首个Tile，S-R
  // 整个S(TileA)(tileM,tileK)，又按照mma_atom的粒度, 又划分为多个小ik。进行二级流水线迭代
  // 因此这里先开始首个小ik的迭代
  // smem -> reg
  // 首先复制首个Tile中的首个ik （S->R）
  cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik)); // S->R  只针对首个Tile中的首个ik
  cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

  // loop over k: i. load tile, ii. mma
  int ntile = k / kTileK;
#pragma unroll 1
  // 整个K轴的大迭代，一个需要处理ntile个TileA,TileB
  for (int itile = 0; itile < ntile; ++itile) {
    int nk = size<2>(tCrA);   // 小k.对应一个TileA中(tileM,TileK),又按照mma-atom划分的k. Tile内迭代该小k,且以流水线方式

#pragma unroll
    for (int ik = 0; ik < nk; ++ik) {  
      // Tile内迭代该小k. 
      // 此时ik所在的Tile,已经同步好,在S上了
      //    首个大迭代：迭代外同步好了
      //    其他大迭代： 每个大Tile的最后一个ik, 计算时，会同时同步下一个大Tile. 到下一个迭代首次，到这里时，下一个Tile也已经都在S上了
      // 且ik本身，已经复制到R了 
      //   首个ik,由迭代前的copy复制到r了;
      //   后续的ik,由上个小迭代的S->R准备好了；(由于2级流水线)
      //   下个大迭代的首个ik, 进来前，不仅S同步好了，上个大迭代最后一个ik计算时，也执行了下个大Tile首个ik的S->R
      // 因此当前TIle的当前ik,已经在R上了，可以直接用于计算了

      int ik_next = (ik + 1) % nk; // 用于准备下一ik的S->R / 下个Tile首个ik的S->R

      if (ik == nk - 1) { 
        // 如果是该Tile的最后一个小k,（已经在R了,该Tile占用的share_mem后续可以空出来了）
        // 此时可以准备等下一个G_>S结束，用于处理下一个Tile的(S-》R)(本TIle已经不需要处理S->R了)
        // 这样下一次大迭代，进来时，整个新Tile,又已经在S上了。可以直接计算了
        // 本时刻只处理该小k的计算，(本Tile的S-R已结束)
        cp_async_wait<kStage - 2>();  // 等下一个Tile复制到S上。下一时刻可以处理新Tile的S->R了
        __syncthreads();
        ismem_read = (ismem_read + 1) % kStage; // S->R时，下个TIle在S中的源位置+1.  S->R读的slot+1 (同样对stage取模)
      }

      // 提前进行本TIle,下个小k的S->R. 同时完成本小k的计算 
      //   写到本Tile该小k对应的寄存器中    
      // 如果此时ik是本TIle的最后一个ik:
      //   本Tile已经不需要s->R了
      //   就读下个Tile的首个小k的S-R(上边已经同步过了):本TIle最后一个小k计算时，提前完成下个TIle首个小k的S->R
      //   写到下个Tile的首个小k对应的寄存器中（永远只有nk个位置）
      // shm -> reg 
      // s[itile][ik + 1] -> r[ik + 1]
      cute::copy(s2r_tiled_copy_a, 
                 tAsA(_, _, ik_next, ismem_read),  // 本来读的是share_mem中，本Tile(ismem_read)的下一个小k
                                                   // 如果ik是本TIle的最后一个ik，就读下个Tile的首个小k。本TIle最后一个小k计算时，提前完成下个TIle首个小k的S->R
                 tCrA_view(_, _, ik_next));        // 写到本Tile该小k对应的寄存器中/ 下个Tile的首个小k对应的寄存器中永远只有nk个位置
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                 tCrB_view(_, _, ik_next));

      if (ik == 0) {
        // 是本Tile的首个ik
        //（说明此时，Tile已经完整复制到share_mem了，一条G->S指令已经执行结束了）
        // 可以新增一条G->S指令，开始异步获取下一个Tile
        if (itile_to_read < ntile) {   // K轴tile没有全部发射完。 itile_to_read:对应G的K轴中，已经发射了G->S指令的Tile id(这是还未处理的)
          // K轴tile没有全部发射完。之前处理到了itile_to_read-1位置
          // 现在可以发射itile_to_read位置对应的TIle,对应的 (G->S)指令。
          cute::copy(g2s_tiled_copy_a, 
                     tAgA_copy(_, _, _, itile_to_read), // K轴 k=[itile_to_read]对应的Tile
                     tAsA_copy(_, _, _, ismem_write));  // 写入share_mem中， slot为ismem_write的位置。ismem_write对n_stage取余，是下个Tile待写入的位置
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));

          ++itile_to_read;  // 记录G中k轴 Tile的读取位置。不取余
          ismem_write = (ismem_write + 1) % kStage; // 记录该Tile写入share_mem的位置。S->R时从该位置读取（对n_stage取余）
        }
        cp_async_fence();
      }

      // 计算本小k，完成一个mma
      // 对应TileA, 沿k轴拓展切分后，ik对应的一次tilemma(含M轴的全部mma)
      // 计算完成后， 结果累加到D对应的寄存器变量tCrD中
      // tCrD: 内存在寄存器上，但是是直接对本cudablock要处理的TileC的划分（对应到本线程）（含M,N维的拓展）（首维对应一次mma_atom的小C结果）
      //      本次只累积一个ik的计算结果：(TIleM,ik),(ik,TIleN）。
      //      内循环累积一次大Tile的计算结果：(TIleM,TIleK),(TileK,TIleN）
      //      外循环累积K轴多次大Tile的计算结果： sumK：(TIleM,TIleK),(TileK,TIleN）
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    }  // for ik   计算完K轴一个Tile,对应的新的Tiledmma
  }    // itile    计算完K轴所有Tile,结果都累积到寄存器tCrD上。
       //          tCrD,和其他线程的寄存器一起，对应(每次mma_atom MileM,TileN和在M,N维拓展的结果_。且沿K轴（拆分为ik轴）计算，积累值
       //                其中在M,N维的拓展，对应TileC的划分，以单次mma_atom对应的小C块，为粒度，沿M,N维度拓展。之后纯累积
  // 本cudablock,得到TileC的完整结果（对应的寄存器变量）


  // 尾阶段高效 （写出到Global_mem）(如果后边有其他算子，也可以放r中不写出)
  // use less shared memory as a scratchpad tile to use large wide instuction
  // Dreg -> shm -> reg -> global

  // 此时TilcC已经全部计算完。因此可以复用原来的share_mem,把本cudablock负责的TileC写出
  auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{}); // 底层内存和AB复用一个。但layout不同。同样是swizzle后的

  // R->S的layout
  // 此时R上放的是完整的TileC。对应新tilemma计算完的大小。（以该TileC对应的所有mma_atom得到的小C为粒度，沿M,N拓展）
  // 首先对atom拓展，得到和新tilemma一样大小的复制区域（即tileC）
  auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);// 直接复制的指令。
  auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx); 
  // 对Tilc，进行划分
  //以mma_atom指令，对应的小C维粒度，沿M,N维度拓展。（以复制完整个tileC）
  //但是寄存器数据已有，只需要对TileC,retile即可
  auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);   // (CPY, CPY_M, CPY_N)
  // 对应的share_mem上的layout, 对应swizzle后的share_mem.(和AB复用一个，但layout不同)
  auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);  // (CPY, _1, _1, pipe)

  // G->S
  S2GCopyC s2g_tiled_copy_c;
  auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
  // tileC大小的share_mem, 按S2GCopyC对应的layout拆分。每个线程可以读连续128个元素。
  // 作为源的S,TIleC(128,128),拆分。以S2GCopyC(32,32)为粒度
  auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);  // (CPY, _1, _1, pipe)
  // 写入到G,对应的G中layout。以S2GCopyC(32,32)为粒度
  auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);  // (CPY, CPY_M, CPY_N)

  auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);  // (CPY_, CPY_MN)
  auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);  // (CPY_, CPY_MN)  mn变成一个维度？按一维迭代

  int step = size<3>(tCsC_r2s);  // pipe. 是kSmemLayoutCBatch，对应写出时，每几个mma_atom复制到S后，可以执行一次G->S. 然后再R->S
#pragma unroll
  for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
    // 迭代每个mma_atom, 对应小C数据的R->S。 mn变一个维度，按1D迭代
    // reg -> shm
#pragma unroll
    for (int j = 0; j < step; ++j) {
      // R-S (复制到不同的stage上？)
      // we add a temp tensor to cope with accumulator and output data type difference
      auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
      cute::copy(tCrC_r2sx(_, i + j), t);   // R->t

      cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));// t-> S (复制到不同的stage上？)
    }
    __syncthreads();
    // 复制完step个mma_atom对应的R, 到S

#pragma unroll
    // shm -> global
    // step个mma_atom的数据，已经写入S.
    // 这里，依次复制这些step.每个step的内容，到G中MN对应的块位置
    for (int j = 0; j < step; ++j) {
      cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    }
    __syncthreads();
    // 复制完step个mma_atom对应的S, 到G. 回到迭代继续复制，直到复制完TileC
  }
}

namespace config {

using namespace cute;

// 用来简单设置config。含各种layout
template <typename T_, int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 32,
          int kStage_ = 5, int kSmemLayoutCBatch_ = 2,
          typename ComputeType = T_>
struct GemmConfig {
  using T = T_;

  // tile configuration
  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kStage = kStage_;
  static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;

  static constexpr int kShmLoadSwizzleM = 3;
  static constexpr int kShmLoadSwizzleS = 3;
  static constexpr int kShmLoadSwizzleB = 3;

  // share_mem结构本身，对应的atom
  // 对于上游的G的读取: cp.sync
  //   128个线程，(32,4)个线程，行主序排列。
  //   每个线程，读取沿行排列的连续的8个元素; 一行4个线程，可以读取逻辑矩阵中一行32个元素。
  //   32行，128个线程，一次可以读取A中(32,32)的逻辑块.
  //   整个G->S的copy, 以对G中该(32,32)的source的读取拓展，拓展到整个TileA(128,32)
  //   (作为G->S的Source)
  //   其中每32个线程，读取的是Ga中（8，32）的逻辑块。每4个线程，读取一行
  // 对于下游的R的写入：ldmatrix*.x4
  //    S2RCopyAtomA: 32个线程，每个线程读取share_mem中，一个连续的8元素(4bank)。
  //    每8个线程，读取的64个元素， 对应G中逻辑上的(8,8)矩阵。可以交给一个mma_atom计算
  //    32个线程，读取的4个(8，8)元素，对应4个mma_atom
  //    (作为S->R的destination)
  // share_mem本身的layout
  //     作为G->S的destination和S->R的source layout. partition后，用于复制（对应sA）
  // share_mem结构本身的atom: 
  //    以对G中逻辑矩阵中(8，32)的块的读取为atom,对应32个线程
  using SmemLayoutAtom = decltype(composition(
      Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},// BMS:333 =888
      make_layout(make_shape(Int<8>{}, Int<kTileK>{}),    // (8,32):(32,1):行主序   M=8，K=32的块
                  make_stride(Int<kTileK>{}, Int<1>{}))));

  // 拓展到TileA(128,32)，对应的share_mem. 流水线数目也拓展
  using SmemLayoutA = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    // share_mem中，需要放n_stage个TileA。
                    // 需要把SmemLayoutAtom atom,拓展到该shARE_mem大小。
                    // 作为后续G->S的目的layout,也作为S->R的源layout
                    make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
  using SmemLayoutB = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

  // 计算部分
  // mma_atom是 16816，对应A(16,16),B(16,8),C(16,8) 用32个线程计算。 每个线程计算8个A元素，4个B元素，4个C元素 
  //                  partitionABC后：首维（A的MMA是(2,2,2)，B的MMA是(2,2)，C的MMA是(2,2)）
  //                  对应指令mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  static constexpr int kMmaEURepeatM = 2;
  static constexpr int kMmaEURepeatN = 2;
  static constexpr int kMmaEURepeatK = 1;

  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});  // atom拓展后, 最终tilemma对应shape（含value拓展）
  static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

  // mma_atom拓展：
  // thread拓展： MNK:（2，2，1）,可以处理(32,16,16)的tiledMMA. 线程数目同样翻4倍，对应128个线程
  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  // value拓展：（1，2，1），使得拓展后的tileshape是该shape(32,32,16)
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
  // tileMMA:(32,32,16),用于一次tilemma计算，对应多次mma. 线程value映射固定 128个线程，对应所有value
  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

  // G-》S
  // G->S atom
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>; // cp.asyc指令，每个线程复制连续128bits（8个fp16元素）(4个bank==16字节)
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;            // T是fp16. ValueLayout对应每个线程读128bits,沿ValueLayout方向读取T类型元素
  // G->S tildCopy
  // 拓展G->S atom,
  // G2SCopyA：把128个线程，分成(32,4)行主序排列，每个线程沿行读取8个元素。一次128个线程，一次可以读取G中(32,32)的A元素块
  // G中的一个tileA(128,32). 需要执行4次G2SCopyA单元(32,32)，读完TileA，到share_mem
  using G2SCopyA =
      decltype(make_tiled_copy(g2s_copy_atom{},
                              // 线程拓展，按照行主序排列128个线程(T0-T4,T5-T8,...T127). 可以处理128*8个元素
                              // 32行，每行4个线程，
                              // 每行4个线程，每个线程沿着行方向，读取8个元素，每行读取32个元素
                              // 因此32行，可以读取A中(32,32)大小的块
                               make_layout(make_shape(Int<32>{}, Int<4>{}),  // 拓展线程.按行主序，排布128个线程 （32,4）
                                           make_stride(Int<4>{}, Int<1>{})),
                              // 指定每个线程，每次执行，沿行读取8个连续元素 （组成的128bits）
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));// value拓展，每个线程沿行复制8个half_t=128bits
  using G2SCopyB = G2SCopyA;

  // S->R: shared memory to register copy
  // S->R atom
  // ldmatrix指令  *x4.  
  //    32个线程:
  //    每8个连续线程：
  //        每个线程读取8个连续元素（share）-> 读取64个元素，对应逻辑上的一个88矩阵块，与A中原始的一个88块对应
  //        读取的64个元素，写入32个线程的寄存器。对应一次mma_atom这8个线程的输入。（一次mma_atom A需要16*16）
  //    4个8线程：
  //        读取4个88矩阵块，对应逻辑上一个(16,16)的矩阵块。
  //        每个88矩阵块，位于32个线程中，不同的寄存器中。
  //        32个线程，提供的4个88矩阵，对应的(16,16)大小，正好对应(一个mma_atom，32个线程，A需要提供的大小(16,16))
  using s2r_copy_op = SM75_U32x4_LDSM_N;     // ldmatrix指令  *x4.  32个线程，每8个连续线程，读取8个连续元素（share）-> 写入32个线程寄存器
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;

  using S2RCopyAtomA = s2r_copy_atom;
  using S2RCopyAtomB = s2r_copy_atom;

  // epilogue: register to global via shared memory
  // 从寄存器，写出C，到global_mem,同样经过share_mem,且经过swizzle

  // epilogue写出C时，
  // share_mem的Atom: 
  // 对应tiledmma拓展后的（M,N）,且是寄存器swizzle后的
  using SmemLayoutAtomC = decltype(composition(
      Swizzle<2, 3, 3>{}, make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                                      make_stride(Int<kMmaPN>{}, Int<1>{}))));
  // share_mem对应的layout
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

  static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >=
                    size(SmemLayoutC{}),
                "C shared memory request is large than A's one pipe");
  // R->S 所用的指令。复制后，destination按上述 swizzle后的layout, 写入对应bank
  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;// atom是直接复制的指令. 1:1复制


  // epilogue写出C时
  // share_mem->G时，对应的atom.
  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>; // atom是 128bits，直接复制的指令， 每个线程可以访存更大位宽
  // share_mem->G时，对应的layout
  using S2GCopyC =
      decltype(make_tiled_copy(S2GCopyAtomC{}, // 每个线程读连续8个元素（沿着行）
                               make_layout(make_shape(Int<32>{}, Int<4>{}), // 128个线程，是(32,4)的thread layout。可以读(32,32)的块
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));

  static constexpr int kThreadNum = size(MMA{}); // 128个线程？对应一次tiledMMA的线程数目

  // 使用的共享内存大小： （TIle）*n_stage * 2(AB)
  static constexpr int shm_size_AB =
      cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});

  // 用于C的写出，作为中介
  static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});

  static constexpr int kShmSize =
      cute::max(shm_size_AB, shm_size_C) * sizeof(T);
};

}  // namespace config

int main(int argc, char *argv[]) {
  using T = cute::half_t;
  using namespace cute;
  using X = Underscore;

  srand(10086);

  cublasHandle_t handle;
  cublasCreate(&handle);
  int cublas_version;
  cublasGetVersion_v2(handle, &cublas_version);
  printf("cuBLAS version: %d\n", cublas_version);

  // default;
  int M = 81920;
  int N = 256;
  int K = 256;

  int enable_cpu = 0;
  int enable_cublaslt = 1;
  int nt = 11;

  using ComputeType = T;

  T *Aptr;
  T *Bptr;
  T *Dptr;             // 放我们的device指针
  T *Dptr_cublas;
  T *Dptr_cublaslt;

  T *Aptr_host;
  T *Bptr_host;
  T *Dptr_host;
  T *Dptr_host_cpu;
  T *Dptr_host_blas;
  T *Dptr_host_cublaslt;

  Aptr_host = (T *)malloc(sizeof(T) * M * K);
  Bptr_host = (T *)malloc(sizeof(T) * N * K);
  Dptr_host = (T *)malloc(sizeof(T) * M * N);

  Dptr_host_cpu = (T *)malloc(sizeof(T) * M * N);
  Dptr_host_blas = (T *)malloc(sizeof(T) * M * N);
  Dptr_host_cublaslt = (T *)malloc(sizeof(T) * M * N);

  cudaMalloc(&Aptr, sizeof(T) * M * K);
  cudaMalloc(&Bptr, sizeof(T) * N * K);
  cudaMalloc(&Dptr, sizeof(T) * M * N);
  cudaMalloc(&Dptr_cublas, sizeof(T) * M * N);
  cudaMalloc(&Dptr_cublaslt, sizeof(T) * M * N);

  // host端的A,B,C
  auto tA = make_tensor(Aptr_host, make_shape(M, K), make_stride(K, 1)); // K major (行主序)   哪一维Major,哪一维stride==1
  auto tB = make_tensor(Bptr_host, make_shape(N, K), make_stride(K, 1)); // K major (行主序)  （K,N）角度看，列主序
  auto tD = make_tensor(Dptr_host, make_shape(M, N), make_stride(N, 1)); // N major (行主序)

  cpu_rand_data(&tA);// 初始化
  cpu_rand_data(&tB);

  clear(tD);

  // 复制到device上
  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * N * K, cudaMemcpyHostToDevice);
  cudaMemcpy(Dptr, Dptr_host, sizeof(T) * M * N, cudaMemcpyHostToDevice);
  cudaMemset(Dptr_cublas, 0, sizeof(T) * M * N);
  cudaMemset(Dptr_cublaslt, 0, sizeof(T) * M * N);

  CublasLtGemm<T, ComputeType> cublaslt_gemm;
  if (enable_cublaslt) {
    cublaslt_gemm.init(Dptr_cublaslt, Bptr, Aptr, N, M, K);
  }

  // 设置每个cudaBlock处理的Tile，对应的MNK<128,128,32>. 流水线3级
  config::GemmConfig<T, 128, 128, 32, 3> gemm_config;

  print(typename decltype(gemm_config)::MMA{});

  // 每个cudablock，分配128个线程，对应tilemma所需的线程数目。其他方向的拓展，仍用这128个线程
  // 原来每个tilemma处理(323216)，但我们每个cudablock要处理（128，128，32）
  // 因此不同方向mma_atom的拓展，综合tilemma+tile的拓展，变成新的tilemma，来处理整个tile。32个线程指令以mma_atom为粒度
  dim3 block = gemm_config.kThreadNum;
  // 对C,进行(M,N)划分，每个cudablock.处理tile大小（128，128）
  dim3 grid((N + gemm_config.kTileN - 1) / gemm_config.kTileN,
            (M + gemm_config.kTileM - 1) / gemm_config.kTileM);

  // 设置共享内存的大小
  int shm_size = gemm_config.kShmSize;

  half alpha = 1.f;
  half beta = 0.f;

  for (int it = 0; it < nt; ++it) {
    // blas，不用管
    cudaMemset(Dptr_cublas, 0, sizeof(T) * M * N);
    cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                                     &alpha, (half *)Bptr, K, (half *)Aptr, K,
                                     &beta, (half *)Dptr_cublas, N);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      printf("cublas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
    }

    if (enable_cublaslt) {
      cudaMemset(Dptr_cublaslt, 0, sizeof(T) * M * N);
      cublaslt_gemm.run();
    }

    // multi-stage
    cudaMemset(Dptr, 0, sizeof(T) * M * N);
    cudaFuncSetAttribute(gemm_multi_stage<decltype(gemm_config)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    // 真正调用kernel做计算
    gemm_multi_stage<decltype(gemm_config)>
        // 每个block算一个Tile。传入g上的指针
        <<<grid, block, shm_size>>>(Dptr, Aptr, Bptr, M, N, K);
  }

  // 结果写回，写回cpu
  cudaMemcpy(Dptr_host, Dptr, sizeof(T) * M * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(Dptr_host_blas, Dptr_cublas, sizeof(T) * M * N,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(Dptr_host_cublaslt, Dptr_cublaslt, sizeof(T) * M * N,
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  printf("block = (%d, %d), gird = (%d, %d), shm = %d\n", block.x, block.y,
         grid.x, grid.y, shm_size);

  if (err == cudaSuccess) {
    printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
  } else {
    printf_fail("err = %d, str = %s\n", err, cudaGetErrorString(err));
  }

  // 结果比较
  // 和cublas比较
  gpu_compare(Dptr, Dptr_cublas, M * N);

  if (enable_cublaslt) {
    gpu_compare(Dptr, Dptr_cublaslt, M * N);
  }

  // 和cpu比较
  auto tD_host = make_tensor(Dptr_host, make_shape(M, N), make_stride(N, 1));
  auto tD_host_cpu =  // cpu的参考结果
      make_tensor(Dptr_host_cpu, make_shape(M, N), make_stride(N, 1));
  auto tD_host_blas =
      make_tensor(Dptr_host_blas, make_shape(M, N), make_stride(N, 1));
  auto tD_host_cublaslt =
      make_tensor(Dptr_host_cublaslt, make_shape(M, N), make_stride(N, 1));

  if (enable_cpu) {
    cpu_gemm(&tD_host_cpu, tA, tB);
    cpu_compare(tD_host_cpu, tD_host, 0.1f);// 和cpu的结果比较
  }

  auto tile = make_tile(min(8, M), min(8, N));
  auto t32x32 = local_tile(tD_host, tile, make_coord(0, 0));  // 可以打印我们实现中。其中的一个32*32的tile.
  auto t32x32_cpu = local_tile(tD_host_cpu, tile, make_coord(0, 0));
  auto t32x32_blas = local_tile(tD_host_blas, tile, make_coord(0, 0));
  auto t32x32_cublaslt = local_tile(tD_host_cublaslt, tile, make_coord(0, 0));

  printf("M = %d, N = %d, K = %d\n", M, N, K);

  printf("our-impl:\n");
  print_tensor(t32x32);
  if (enable_cpu) {
    printf("cpu:\n");
    print_tensor(t32x32_cpu);
  }
  printf("cublas:\n");
  print_tensor(t32x32_blas);

  if (enable_cublaslt) {
    printf("cublaslt:\n");
    print_tensor(t32x32_cublaslt);
  }
}

// ./gemm-multi-stage
