#include <omp.h>
#include <memory.h>
#include <inaccel/coral>
namespace Bn128 {
    static void init();
}
#include "bn128.hpp"
#include "misc.hpp"
/*
template <typename Curve>
void ParallelMultiexp<Curve>::initAccs() {
    #pragma omp parallel for
    for (int i=0; i<nThreads; i++) {
        memset((void *)&(accs[i*accsPerChunk]), 0, accsPerChunk*sizeof(PaddedPoint));
    }
}
*/

template <typename Curve>
void ParallelMultiexp<Curve>::initAccs() {
    #pragma omp parallel for
    for (int i=0; i<nThreads*accsPerChunk; i++) {
        g.copy(accs[i].p, g.zero());
    }
}

template <typename Curve>
uint32_t ParallelMultiexp<Curve>::getChunk(uint32_t scalarIdx, uint32_t chunkIdx) {
    uint32_t bitStart = chunkIdx*bitsPerChunk;
    uint32_t byteStart = bitStart/8;
    uint32_t efectiveBitsPerChunk = bitsPerChunk;
    if (byteStart > scalarSize-8) byteStart = scalarSize - 8;
    if (bitStart + bitsPerChunk > scalarSize*8) efectiveBitsPerChunk = scalarSize*8 - bitStart;
    uint32_t shift = bitStart - byteStart*8;
    uint64_t v = *(uint64_t *)(scalars + scalarIdx*scalarSize + byteStart);
    v = v >> shift;
    v = v & ( (1 << efectiveBitsPerChunk) - 1);
    return uint32_t(v);
}

template <typename Curve>
void ParallelMultiexp<Curve>::processChunk(uint32_t idChunk) {
    #pragma omp parallel for
    for(uint32_t i=0; i<n; i++) {
        if (g.isZero(bases[i])) continue;
        int idThread = omp_get_thread_num();
        uint32_t chunkValue = getChunk(i, idChunk);
        if (chunkValue) {
            g.add(accs[idThread*accsPerChunk+chunkValue].p, accs[idThread*accsPerChunk+chunkValue].p, bases[i]);
        }
    }
}

template <typename Curve>
void ParallelMultiexp<Curve>::packThreads() {
    #pragma omp parallel for
    for(uint32_t i=0; i<accsPerChunk; i++) {
        for(uint32_t j=1; j<nThreads; j++) {
            if (!g.isZero(accs[j*accsPerChunk + i].p)) {
                g.add(accs[i].p, accs[i].p, accs[j*accsPerChunk + i].p);
                g.copy(accs[j*accsPerChunk + i].p, g.zero());
            }
        }
    }
}

template <typename Curve>
void ParallelMultiexp<Curve>::reduce(typename Curve::Point &res, uint32_t nBits) {
    if (nBits==1) {
        g.copy(res, accs[1].p);
        g.copy(accs[1].p, g.zero());
        return;
    }
    uint32_t ndiv2 = 1 << (nBits-1);


    PaddedPoint *sall = new PaddedPoint[nThreads];
    memset(sall, 0, sizeof(PaddedPoint)*nThreads);

    typename Curve::Point p;
    #pragma omp parallel for
    for (uint32_t i = 1; i<ndiv2; i++) {
        int idThread = omp_get_thread_num();
        if (!g.isZero(accs[ndiv2 + i].p)) {
            g.add(accs[i].p, accs[i].p, accs[ndiv2 + i].p);
            g.add(sall[idThread].p, sall[idThread].p, accs[ndiv2 + i].p);
            g.copy(accs[ndiv2 + i].p, g.zero());
        }
    }
    for (u_int32_t i=0; i<nThreads; i++) {
        g.add(accs[ndiv2].p, accs[ndiv2].p, sall[i].p);
    }

    typename Curve::Point p1;
    reduce(p1, nBits-1);

    for (u_int32_t i=0; i<nBits-1; i++) g.dbl(accs[ndiv2].p, accs[ndiv2].p);
    g.add(res, p1, accs[ndiv2].p);
    g.copy(accs[ndiv2].p, g.zero());
    delete[] sall;
}

template <typename Curve>
void ParallelMultiexp<Curve>::multiexp(typename Curve::Point &r, typename Curve::PointAffine *_bases, uint8_t* _scalars, uint32_t _scalarSize, uint32_t _n, uint32_t _nThreads) {
    nThreads = _nThreads==0 ? omp_get_max_threads() : _nThreads;
//    nThreads = 1;
    bases = _bases;
    scalars = _scalars;
    scalarSize = _scalarSize;
    n = _n;

    if (n==0) {
        g.copy(r, g.zero());
        return;
    }
    if (n==1) {
        g.mulByScalar(r, bases[0], scalars, scalarSize);
        return;
    }
    bitsPerChunk = log2(n / PME2_PACK_FACTOR);
    if (bitsPerChunk > PME2_MAX_CHUNK_SIZE_BITS) bitsPerChunk = PME2_MAX_CHUNK_SIZE_BITS;
    if (bitsPerChunk < PME2_MIN_CHUNK_SIZE_BITS) bitsPerChunk = PME2_MIN_CHUNK_SIZE_BITS;
    nChunks = ((scalarSize*8 - 1 ) / bitsPerChunk)+1;
    accsPerChunk = 1 << bitsPerChunk;  // In the chunks last bit is always zero.

    typename Curve::Point *chunkResults = new typename Curve::Point[nChunks];
    accs = new PaddedPoint[nThreads*accsPerChunk];
    // std::cout << "InitTrees " << "\n"; 
    initAccs();

    for (uint32_t i=0; i<nChunks; i++) {
        // std::cout << "process chunks " << i << "\n"; 
        processChunk(i);
        // std::cout << "pack " << i << "\n"; 
        packThreads();
        // std::cout << "reduce " << i << "\n"; 
        reduce(chunkResults[i], bitsPerChunk);
    }

    delete[] accs;

    g.copy(r, chunkResults[nChunks-1]);
    for  (int j=nChunks-2; j>=0; j--) {
        for (uint32_t k=0; k<bitsPerChunk; k++) g.dbl(r,r);
        g.add(r, r, chunkResults[j]);
    }

    delete[] chunkResults; 
}

template <typename Curve>
void InAccelMultiexp<Curve>::multiexpG1(typename Curve::Point &r, typename Curve::PointAffine *_bases, uint8_t* _scalars, uint32_t _scalarSize, uint32_t _n, uint32_t _nThreads) {
    assert(_scalarSize == 32);

    size_t length = _n;

    inaccel::vector<uint8_t> vec_buf(length * 64);
    inaccel::vector<uint8_t> scalar_buf(length * 32);
    inaccel::vector<uint8_t> result_buf(96);

    Bn128::init();

    size_t i;
    for (i = 0; i < length; i++) {
        if (g.isZero(_bases[i])) {
            continue;
        }

        mpz_t vec_X_mpz, vec_Y_mpz;
        mpz_inits(vec_X_mpz, vec_Y_mpz, NULL);
        g.F.toMpz(vec_X_mpz, _bases[i].x);
        g.F.toMpz(vec_Y_mpz, _bases[i].y);

        Bn128::af_p_t<Bn128::f_t<1>> tmp;
        mpz_set(tmp.x.c[0], vec_X_mpz);
        mpz_set(tmp.y.c[0], vec_Y_mpz);
        Bn128::af_export(&vec_buf[i * 64], Bn128::to_mont(tmp));

        mpz_clears(vec_X_mpz, vec_Y_mpz, NULL);

        memcpy(&scalar_buf[i * 32], &_scalars[i * 32], 32);
    }

    if (i == 0) {
        return;
    }

    length = ((i - 1) | 7) + 1;

    vec_buf.resize(length * 64);
    scalar_buf.resize(length * 32);

    inaccel::request multiexp("libff.multiexp.alt-bn128-g1");
    multiexp.arg(length).arg(vec_buf).arg(scalar_buf).arg(result_buf);
    inaccel::submit(multiexp).get();

    Bn128::jb_p_t<Bn128::f_t<1>> result_jacobian;
    Bn128::jb_import(result_jacobian, result_buf.data());
    Bn128::af_p_t<Bn128::f_t<1>> result_affine = Bn128::mont_jb_to_af(result_jacobian);

    typename Curve::PointAffine result;
    g.F.fromMpz(result.x, result_affine.x.c[0]);
    g.F.fromMpz(result.y, result_affine.y.c[0]);
    g.copy(r, result);
}

template <typename Curve>
void InAccelMultiexp<Curve>::multiexpG2(typename Curve::Point &r, typename Curve::PointAffine *_bases, uint8_t* _scalars, uint32_t _scalarSize, uint32_t _n, uint32_t _nThreads) {
    assert(_scalarSize == 32);

    size_t length = _n;

    inaccel::vector<uint8_t> vec_buf(length * 128);
    inaccel::vector<uint8_t> scalar_buf(length * 32);
    inaccel::vector<uint8_t> result_buf(192);

    Bn128::init();

    size_t i;
    for (i = 0; i < length; i++) {
        if (g.isZero(_bases[i])) {
            continue;
        }

        mpz_t vec_X_c0_mpz, vec_X_c1_mpz, vec_Y_c0_mpz, vec_Y_c1_mpz;
        mpz_inits(vec_X_c0_mpz, vec_X_c1_mpz, vec_Y_c0_mpz, vec_Y_c1_mpz, NULL);
        g.F.F.toMpz(vec_X_c0_mpz, _bases[i].x.a);
        g.F.F.toMpz(vec_X_c1_mpz, _bases[i].x.b);
        g.F.F.toMpz(vec_Y_c0_mpz, _bases[i].y.a);
        g.F.F.toMpz(vec_Y_c1_mpz, _bases[i].y.b);

        Bn128::af_p_t<Bn128::f_t<2>> tmp;
        mpz_set(tmp.x.c[0], vec_X_c0_mpz);
        mpz_set(tmp.x.c[1], vec_X_c1_mpz);
        mpz_set(tmp.y.c[0], vec_Y_c0_mpz);
        mpz_set(tmp.y.c[1], vec_Y_c1_mpz);
        Bn128::af_export(&vec_buf[i * 128], Bn128::to_mont(tmp));

        mpz_clears(vec_X_c0_mpz, vec_X_c1_mpz, vec_Y_c0_mpz, vec_Y_c1_mpz, NULL);

        memcpy(&scalar_buf[i * 32], &_scalars[i * 32], 32);
    }

    if (i == 0) {
        return;
    }

    length = ((i - 1) | 7) + 1;

    vec_buf.resize(length * 128);
    scalar_buf.resize(length * 32);

    inaccel::request multiexp("libff.multiexp.alt-bn128-g2");
    multiexp.arg(length).arg(vec_buf).arg(scalar_buf).arg(result_buf);
    inaccel::submit(multiexp).get();

    Bn128::jb_p_t<Bn128::f_t<2>> result_jacobian;
    Bn128::jb_import(result_jacobian, result_buf.data());
    Bn128::af_p_t<Bn128::f_t<2>> result_affine = Bn128::mont_jb_to_af(result_jacobian);

    typename Curve::PointAffine result;
    g.F.F.fromMpz(result.x.a, result_affine.x.c[0]);
    g.F.F.fromMpz(result.x.b, result_affine.x.c[1]);
    g.F.F.fromMpz(result.y.a, result_affine.y.c[0]);
    g.F.F.fromMpz(result.y.b, result_affine.y.c[1]);
    g.copy(r, result);
}
