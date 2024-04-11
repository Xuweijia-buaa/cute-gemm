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

  // 实例化一个tiled_mma, 可以一次处理32_32_16的矩阵乘
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  // 对tileMMA来说， 
  // 每个MMAAtom 16*8*16中，对应32个线程，
  //    A矩阵（M,k）：
  //    32个线程，处理(16*16)的A矩阵
  //    每个线程，可以处理(16,16)的A矩阵中的8个元素（对应一定的layout）  
  // tileMMA:
  //    Atom拓展：在M方向拓展2，N方向拓展2，但线程也拓展。每个线程处理的线程和逻辑layout不变
  //    Value拓展：只在N方向拓展，因此每个线程对应的线程和逻辑layout也不变
  // 对一次tileMMA来说：
  //    一次tilemma计算，对应一次32*32*16的矩阵乘。（128个线程）
  //    需要提供的A大小是（32，16）    （64个线程）
  //    其中每个线程处理A中8个元素,对应一定的layout    ~~

  // 以A为例：
  // 本cudablock，需要处理是a大小是：gA(kTileM=128, kTileK=32, num_tile_k),对应K轴的所有块
  //             A中每个分块大小是(kTileM=128,kTileK=32),
  // 需要(4,2)次tilemma计算，才能算完. 每次算(32,16)

  // partition_A，对gA的前2维(128,32)进行划分,提供该线程需要读取到寄存器中的数据
  // 首个维度是本线程，执行一次tiledMMA需要处理的元素(2,2,2)数目，对应一次tiledMMA计算
  //    The first mode, `MMA`, of the result tensors hold all of the elements that a single instruction will consume
  // 第2，3个维度，对应ga的前2为对应的块(128,32),共需要几次MMA计算。
  //    这里gA的前2维(128,32), 一共需要(4,2)次mma计算.
  // 因此最终得到的tgaA的前3个维度是：
  //    （(2,2,2),4,2）
  //     代表本线程每次提供8个元素，进行一次tilemma计算（对应一次323216的矩阵乘），共需要沿(M=4,K=2)方向读取8次，才能计算一个完整的Ak块。
  //     即ga(tileM,tileK,*) -> (thread_single_MMA_element,MMa_num_M, MMA_num_K,*), 拆成多次tiileMMA计算
  //     只对ga Tensor的前两维进行划分(M,K.*)
  // 划分成3个维度（本线程一次喂给MMA计算所需要的元素（对应的layout），在M维重复次数，在K维重复次数），对应tiledMMA,
  // (MMA, MMA_M, MMA_K, num_tile_k)->
  //（单次MMA本线程需提供的元素（对应的layout），在M维重复次数，在K维重复次数(以计算多个mma,算完ga前2维对应的块）, ga所含的其他维度）
  // ((2,2,2),4,2,8)
  auto tAgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K, num_tile_k)->（单次MMA计算本线程需提供的元素（对应的layout），在M维重复次数，在K维重复次数，以计算完整个ga的一个小块）
  // 对B分块，类似，得到算完B中小块Bk,(线程需要提供的数据，和在N维，K维的迭代次数)，以算完一次gB的(kTileN, kTileK)
  auto tBgB = thr_mma.partition_B(gB);  // (MMA, MMA_N, MMA_K, num_tile_k)
  auto tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)
  
  // 上边的数据，仍在global mem中。
  // 但我们迭代时，ga,gb中的小块，每次都放到寄存器上，才能给TensorCore算
  // 把划分的逻辑结果，放到本线程的寄存器上
  // 仍是本线程每个mma操作，（需要提供的(数据)，以及迭代完该小块，对应的M,N需要拓展的维度）
  // 每个线程的寄存器，提供一部分数据，但所有线程一起，提供一次mma所需的数据，进行一次TensorCore上的tilemma操作。
  // M维，N维拓展后，可以迭代完一个（kTileM=128, kTileK=32）大小的块
  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)  slice_ga, 输入shape是（kTileM=128, kTileK=32）
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)
 
  clear(tCrC);// 每个线程的寄存器，清空为0，
  
  int num_tile_k = size<2>(gA);
#pragma unroll 1
  for(int itile = 0; itile < num_tile_k; ++itile) {  
    // 沿着K轴迭代
    // 把本线程global mem中的内容，复制到本线程的寄存器上。(mma,M,N)
    // 128个线程一起，可以完成拓展了M维，N维的MN次tiled MMA计算，算完K轴一个小块
    cute::copy(tAgA(_, _, _, itile), tArA);  // tAgA(_, _, _, itile): k轴的k小块数据。对应的本线程数据(MMA,MMA_M,MMA_K)
    cute::copy(tBgB(_, _, _, itile), tBrB);

    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC); // 算完gc K轴中一个小块： M维，N维的MN次tiled MMA,结果写入tCrC。 每个寄存器放一部分
  }

  // K轴迭代结束后，cudablock中负责的C中该小块，结果正确。结果累积在寄存器中 （D=A*B+C,其中D==C，累加）
  // 复制回global_mem. 完成该cudablock的任务
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

  // 使用的mma指令是： mnk_16816,  即A对应16,16   B对应16*8  C对应16，8 
  // 32个线程。 atom TN: 要求A转置(行主序)，B不转置（以(K,N)看,列主序）
  // 对A: 32个线程，读取16*16个元素，每个线程读取一定分布下的8个元素 （后续N拓展，不影响A的TVlayout）
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;   
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  // tiledMMA
  // 拓展出(2,2,1)，同时拓展对应线程。拓展出128个线程
  // 128个线程，可以计算32_32_16的矩阵乘. 
  using MMA = decltype(make_tiled_mma(mma_atom{}, 
                      // 拓展出4个atom: (2,2,1)，对应128个线程. MN都*2，可以计算32_16_16的矩阵乘。4个warp
                      make_layout(Shape<_2, _2, _1>{}),  
                      // N方向，元素数目拓展2倍，负责更多的数据。
                      // 线程数目不变，直接映射过去。原来32，现在64，可以计算32_32_16的矩阵乘. 每个线程算更多元素
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
  printf("size(MMA{}):%d\n",size(MMA{}));


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

  print_tensor(tc1);
  print_tensor(tc1_cublas);
}

template <typename T>
void gen_rand_data(T *data, int n) {
  for (int i = 0; i < n; ++i) {
    float v = (rand() % 200 - 100) * 0.01;
    data[i] = v;
  }
}


// make
// size(MMA{})