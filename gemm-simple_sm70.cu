#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

template <typename T>
void gen_rand_data(T *data, int n);

template <typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k) {

  using namespace cute;

  // 把传入的原始指针（一维连续内存），转为逻辑上的cute_tensor
  // mk个连续元素，表示为逻辑上的(m,k):(k,1), 行主序。逻辑上行表示的元素连续。不同行stride=k
  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  // nk个连续元素，表示为逻辑上的(n,k):(k:1), 同样行主序。（如果算乘法的话，是转置后的(k:n)。以(K,N)看，（列主序）。指令要求的是以(K,N看)）  
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  // mn个连续元素，用于写入，表示为逻辑上的(m,n):(n,1)， 行主序
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

  // 本cudablock,处理C中（kTileM，kTileN)大小的块，对应的完整矩阵乘
  int ix = blockIdx.x;
  int iy = blockIdx.y;

  int tidx= threadIdx.x;
  // blockDim.x 32,blockDim.y 1 blockDim.z 1 
  // 一个cudablock,只用了个32个线程
  // 每个线程，提供计算该cudablock的每个小块C时，
  //         涉及到的mma_atom的所有数据(对应的寄存器）
  //         每个mma_atom对应一条指令，（M,N）维拓展后，一个小块C的计算，共需要MN个atom指令
  //         对应K轴，一次cute::gemm的计算。
  if (tidx==0 && ix==0 && iy==0){
    printf("blockDim.x %d,blockDim.y %d blockDim.z %d \n",blockDim.x,blockDim.y,blockDim.z);
  }




  // 得到一个cudablock要处理的任务：
  // C中小块：(kTileM, kTileN)
  // 原始矩阵C，用(kTileM,kTileN)模式(左上角)，划分成多个tile. 取出该cudablock对应的tile (对应的逻辑Tensor)
  Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
  // 原始矩阵A,用(kTileM,kTileK)模式，划分成多个tile
  //          取M维度上，该block对应iy行的所有块。对应（TileM,K） 。 此时K轴也被切分，全取，待迭代
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
  // 原始矩阵B(N,K),用(kTileN,kTileK)模式，划分成多个tile
  //          取N维度上，ix个整块。对应（TileN,K）。此时K轴也被切分，全取，待迭代
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));

  //  gA(kTileM, kTileK, num_tile_k)   最后一维K，已经按kTileK分块了
  //  gB(kTileN, kTileK, num_tile_k)   最后一维K，已经按kTileK分块了
  //  gC(kTileM, kTileN)               本cudablock要处理的C小块

  // 实例化一个tiled_mma, 可以处理32_32_16的矩阵乘
  TiledMMA tiled_mma;

  
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  // 从gA(kTileM=128, kTileK=32, num_tile_k)中，抽取该线程负责的块


  // 对ga(128,32),按线程进行分块，
  // 对Tensor的前两维进行划分(M,K.*)
  // 划分成3个维度（本线程一次喂给MMA计算所需要的元素（对应的layout），在M维重复次数，在K维重复次数），对应tiledMMA,
  // (本线程一次喂给MMA计算所需要的元素（对应的layout），在M维重复次数，在K维重复次数,*)
  // partition后，既按照原始块的大小划分，也按照tileMMA划分，拆成mma_atom粒度，和M,K维度的拓展（该线程涉及到的）
  auto tAgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K, num_tile_k)
  // 一次mma atom,需要A(8,4), 需要本线程提供4个元素
  // 一次tileMMA, 本身在M维拓展2倍(拓展线程)，K维不拓展  需要A(16,4)。本线程需要提供4个元素
  // 原始要划分的ga一个块大小为(128,32)，对应M维，K维拓展出的(8,8)个tiledmma，对应(16,8)个atom_mma,但其中一半M，不需要使用本线程
  // 因此本线程每次mma_atom提供4个元素，共(MMA_M=8,MMA_K=8)次。提供所有寄存器对应的元素
  //print(gA);
  // if (tidx==0 && ix==0 && iy==0){
  //   // gmem_ptr[16b](0x7f6558000000) o (_4,_8,_8,8):(_1,4096,_4,_32)
  //   print(tAgA);
  // }
  //print(cute::rank<>(tAgA));
  //print(tAgA);


  auto tBgB = thr_mma.partition_B(gB);  // (MMA, MMA_N, MMA_K, num_tile_k)
  // 总的gb(N=128,K=32)
  // 一次mma atom,需要B(8,4), 其中需要本线程提供4个元素
  // 一次tileMMA, 本身在N维拓展2(拓展线程)*2（拓展value），K维不拓展  需要B(32,4)。本线程需要提供8个元素
  // 总的一块gb（128，32）， 可以沿着N,K维划分为(4,8)个tilemma, 对应(16,8)个mma atom. 其中N维只有拓展value的部分，需要本线程提供元素，拓展线程部分不需要
  // 因此本线程为每次mma_atom提供4个元素，共(MMA_M=8,MMA_K=8)次。可以算完整个gb。
  //print(gB);
  // if (tidx==0 && ix==0 && iy==0){
  //   // gmem_ptr[16b](0x7f226a800000) o (_4,_8,_8,8):(_1,4096,_4,_32)
  //   print(tBgB);
  // }
  //print(tBgB);

  auto tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)
  // 总的gc(128,128)
  // 每个atom,需要C(8,8)。8个线程，每个线程读取一定分布下的8个元素
  // 一次tileMMA, 本身在M(拓展线程2)，N维拓展2(拓展线程)*2（拓展value）。对应C(16,32)。 只有N维度拓展value, 其他MN维度拓展的2，不归本线程
  // 总的gc(128,128),可以划分为(8,4)个tiledmma，对应(2*8，2*2*4)个mma。其中本线程不处理拓展线程的，只处理(8,2*4)个mma
  // 因此是本线程负责计算的mma_atoms，在M维，N维拓展(8,2*4)，既包含tileamma对应的拓展，也包含gc拆分对应的拓展，合成一个大的新tileMMA
  // 只参与该线程负责的计算。每次mma_atom计算8个元素
  //print(gC);
  //print(tCgC);
  // if (tidx==0 && ix==0 && iy==0){
  //   //gmem_ptr[16b](0x7fddec000000) o (_8,_8,_8):(_1,4096,_16)
  //   print(tCgC);
  // }

  // 本线程计算一个小块C时，需要提供的全部数据（对应的寄存器），对应一次cute::gemm的计算。
  // 每次cute::gemm计算，完整一个小块C的完整计算
  // 拆分后，对应多个mma_atom的计算，每个mma_atom对应一条指令
  // 其中每个线程，参与其中部分mma_atom的计算，且在M维N维分别拓展M,N次，对应该线程共参与MN次MMA atom的计算，共同构成一个新的tileMMA计算
  // 这里提供本线程参与该新的tileMMA计算时，需要提供的所有寄存器数据：
  // 含每次mma_atom本线程需要提供的数据，和本线程参与的多次MMA计算(M,N)计算，以共同完成原始块C的计算
  // (本线程一次mma_atom计算需要提供数据，和本线程参与的MMA_atom计算次数，在M维，N维的拓展)。本线程本次cute::gemm计算，共需参与MN次mma_atom计算
  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)  本线程参与每次cute_gemm计算，需要提供的寄存器中的数据，对应多次MMA_atom计算
  // if (tidx==0 && ix==0 && iy==0){
  //   print(tArA);    //ptr[16b](0x7f9ce4fff030) o (_4,_8,_8):(_1,_4,_32)
  // }

  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
  // if (tidx==0 && ix==0 && iy==0){
  //   //ptr[16b](0x7fc4e6fff050) o (_4,_8,_8)：(_1,_4,_32)
  //   print(tBrB);
  // }
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)
  // if (tidx==0 && ix==0 && iy==0){
  //   //ptr[16b](0x7fcdd0fff6b0) o (_8,_8,_8):(_1,_8,_64)
  //   print(tCrC);
  // }
 
  clear(tCrC);
  
  int num_tile_k = size<2>(gA);
#pragma unroll 1
  for(int itile = 0; itile < num_tile_k; ++itile) {  
    // 沿着K轴迭代
    cute::copy(tAgA(_, _, _, itile), tArA);
    cute::copy(tBgB(_, _, _, itile), tBrB);

    //print_tensor(tArA);
    //print_tensor(tBrB);

    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  cute::copy(tCrC, tCgC); 
}

int main() {
  srand(10086);

  using T = cute::half_t;
  using namespace cute;

  T *Cptr;
  T *Aptr;
  T *Bptr;

  int m = 81920;
  int n = 256;
  int k = 256;

  cudaMalloc(&Cptr, sizeof(T) * m * n);
  cudaMalloc(&Aptr, sizeof(T) * m * k);
  cudaMalloc(&Bptr, sizeof(T) * k * n);

  T *Aptr_host;
  T *Bptr_host;
  Aptr_host = (T*)malloc(sizeof(T) * m * k);
  Bptr_host = (T*)malloc(sizeof(T) * n * k);
  gen_rand_data(Aptr_host, m * k);
  gen_rand_data(Bptr_host, n * k);

  // 初始化global mem上的A(m.k),B(kn)
  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * n * k, cudaMemcpyHostToDevice);

  // 使用的mma指令是： mnk_884,  即A对应(84)   B对应(48)  C对应(88)
  // atom: 8个线程TN: 要求A转置(行主序)，B不转置（以(K,N)看,列主序）
  // 对A(8,4): 8个线程，每个线程读取一定分布下的4个元素 （后续N拓展，不影响A的TVlayout）
  // 对B(4,8): 8个线程，每个线程读取一定分布下的4个元素
  // 对C(8,8): 8个线程，每个线程读取一定分布下的8个元素
  // tile后，可以计算16_32_4的矩阵乘。32个线程
  //  对A(16,4): 16个线程，每个线程读取一定分布下的4个元素
  //  对B(4,32): 16个线程，每个线程读取8个元素
  //  对C(16,32):32个线程，每个线程读取16个元素
  //using mma_op = SM80_16x8x16_F16F16F16F16_TN; 
  using mma_op = SM70_8x8x4_F16F16F16F16_TN;      // 变成（8，4）（4，8）-》（8，8）
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  // tiledMMA
  // 拓展出(2,2,1)，同时拓展对应线程。拓展出32个线程
  // 128个线程，可以计算32_32_16的矩阵乘. 
  using MMA = decltype(make_tiled_mma(mma_atom{}, 
                      // 拓展出4个atom: (2,2,1)，对应32个线程. MN都*2，可以计算16_16_4的矩阵乘
                      make_layout(Shape<_2, _2, _1>{}),  
                      // N方向，元素数目拓展2倍，负责更多的数据。
                      // 线程数目不变，直接映射过去。原来16，现在32，可以计算16_32_4的矩阵乘. 每个线程算更多元素
                      make_layout(Shape<_1, _2, _1>{})));// 

  // cudablock粒度分块，每个cudablock,处理一个(kTileM=128,kTileN=128)的C中小块
  // slice-K: K轴按照3kTileK切块，AB沿K轴的块迭代，每次取出第k个小块，计算后累加到本block处理的C中小块上
  constexpr int kTileM = 128; 
  constexpr int kTileN = 128; 
  constexpr int kTileK = 32; 
  // C(m.n), 划分为（kTileM，kTileN)大小的块。
  // 从而根据原来大矩阵C的规模，设定分块数目
  // 每个cudablock,处理C中（kTileM，kTileN)大小的完整矩阵乘
  dim3 grid(n / kTileN, m / kTileM); 

  // 每个block, 分配32_32_16个线程，处理(128,128)大小的C块
  dim3 block(size(MMA{}));
  printf("size(MMA{}):%d\n",size(MMA{})); // 131072


  for (int i = 0; i < 100; ++i) {
    // 调用，传入A,B,C的原始gloabal mem指针
    // 分块大小，指定的MMA。作为模板参数传入
    gemm_simple<T, kTileM, kTileN, kTileK, MMA><<<grid, block>>>(Cptr, Aptr, Bptr, m, n, k);
  }
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  // cublas
  T *Cptr_cublas;

  cudaMalloc(&Cptr_cublas, sizeof(T) * m * n);

  cublasHandle_t handle;
  cublasCreate(&handle);

  half alpha = half(1.f);
  half beta = half(0.f);
  for (int i = 0; i < 100; ++i) {
    cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
          	  n, m, k,
          	  &alpha,
          	  (half *)Bptr, k,
          	  (half *)Aptr, k,
          	  &beta,
          	  (half *)Cptr_cublas, n);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      printf("blas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
    }
  }

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  T *Cptr_host;
  T *Cptr_cublas_host;

  Cptr_host = (T*)malloc(sizeof(T) * m * n);
  Cptr_cublas_host = (T*)malloc(sizeof(T) * m * n);

  // compare
  cudaMemcpy(Cptr_host, Cptr, sizeof(T) * m * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(Cptr_cublas_host, Cptr_cublas, sizeof(T) * m * n, cudaMemcpyDeviceToHost);

  float threshold = 0.1;
  for (int i = 0; i < m * n; ++i) {
    float v1 = Cptr_host[i];
    float v2 = Cptr_cublas_host[i];
    if (fabs(v2 - v1) > threshold) {
      printf("v1 = %f, v2 = %f\n", v1, v2);
    }
  }

  Tensor tensor_C = make_tensor(Cptr_host, make_shape(m, n), make_stride(n, 1));
  Tensor tensor_C_cublas = make_tensor(Cptr_host, make_shape(m, n), make_stride(n, 1));

  auto tile = make_tile(8, 8);
  auto coor = make_coord(0, 0);
  Tensor tc1 = local_tile(tensor_C, tile, coor);
  Tensor tc1_cublas = local_tile(tensor_C_cublas, tile, coor);

  // print_tensor(tc1);
  // print_tensor(tc1_cublas);
}

template <typename T>
void gen_rand_data(T *data, int n) {
  for (int i = 0; i < n; ++i) {
    float v = (rand() % 200 - 100) * 0.01;
    data[i] = v;
  }
}


// make
// ./gemm-simple_sm70