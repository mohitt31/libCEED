/// Microbenchmark: tensor contraction with and without even-odd decomposition
/// Usage: ./bench-even-odd <backend> [ncomp] [n_iter]
/// Example: ./bench-even-odd /cpu/self/avx/blocked 1 10000
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double get_time(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <backend> [ncomp] [n_iter]\n", argv[0]);
    return 1;
  }

  const char *backend = argv[1];
  CeedInt     ncomp   = argc > 2 ? atoi(argv[2]) : 1;
  int         n_iter  = argc > 3 ? atoi(argv[3]) : 5000;
  CeedInt     dim     = 3;
  CeedInt     n_elem  = 1;

  printf("backend=%s ncomp=%d n_iter=%d dim=%d\n", backend, (int)ncomp, n_iter, (int)dim);
  printf("%5s %5s %12s %12s %12s %12s\n", "p", "q", "interp(us)", "grad(us)", "interpT(us)", "gradT(us)");

  for (CeedInt p = 2; p <= 12; p++) {
    CeedInt q = p + 1;

    Ceed      ceed;
    CeedBasis basis;

    CeedInit(backend, &ceed);
    CeedBasisCreateTensorH1Lagrange(ceed, dim, ncomp, p, q, CEED_GAUSS, &basis);

    CeedInt p_dim = 1, q_dim = 1;
    for (CeedInt d = 0; d < dim; d++) {
      p_dim *= p;
      q_dim *= q;
    }

    CeedVector u_interp, v_interp, u_grad, v_grad;

    CeedVectorCreate(ceed, n_elem * ncomp * p_dim, &u_interp);
    CeedVectorCreate(ceed, n_elem * ncomp * q_dim, &v_interp);
    CeedVectorCreate(ceed, n_elem * ncomp * p_dim, &u_grad);
    CeedVectorCreate(ceed, n_elem * ncomp * dim * q_dim, &v_grad);

    CeedVectorSetValue(u_interp, 1.0);
    CeedVectorSetValue(v_interp, 0.0);
    CeedVectorSetValue(u_grad, 1.0);
    CeedVectorSetValue(v_grad, 0.0);

    // Warmup
    for (int i = 0; i < 100; i++) {
      CeedBasisApply(basis, n_elem, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u_interp, v_interp);
      CeedBasisApply(basis, n_elem, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, u_grad, v_grad);
    }

    // Time INTERP forward
    double t0 = get_time();
    for (int i = 0; i < n_iter; i++) {
      CeedBasisApply(basis, n_elem, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u_interp, v_interp);
    }
    double t_interp = (get_time() - t0) / n_iter * 1e6;

    // Time GRAD forward
    t0 = get_time();
    for (int i = 0; i < n_iter; i++) {
      CeedBasisApply(basis, n_elem, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, u_grad, v_grad);
    }
    double t_grad = (get_time() - t0) / n_iter * 1e6;

    // Time INTERP transpose
    CeedVector u_interp_t, v_interp_t;
    CeedVectorCreate(ceed, n_elem * ncomp * q_dim, &u_interp_t);
    CeedVectorCreate(ceed, n_elem * ncomp * p_dim, &v_interp_t);
    CeedVectorSetValue(u_interp_t, 1.0);
    CeedVectorSetValue(v_interp_t, 0.0);

    t0 = get_time();
    for (int i = 0; i < n_iter; i++) {
      CeedBasisApply(basis, n_elem, CEED_TRANSPOSE, CEED_EVAL_INTERP, u_interp_t, v_interp_t);
    }
    double t_interp_t = (get_time() - t0) / n_iter * 1e6;

    // Time GRAD transpose
    CeedVector u_grad_t, v_grad_t;
    CeedVectorCreate(ceed, n_elem * ncomp * dim * q_dim, &u_grad_t);
    CeedVectorCreate(ceed, n_elem * ncomp * p_dim, &v_grad_t);
    CeedVectorSetValue(u_grad_t, 1.0);
    CeedVectorSetValue(v_grad_t, 0.0);

    t0 = get_time();
    for (int i = 0; i < n_iter; i++) {
      CeedBasisApply(basis, n_elem, CEED_TRANSPOSE, CEED_EVAL_GRAD, u_grad_t, v_grad_t);
    }
    double t_grad_t = (get_time() - t0) / n_iter * 1e6;

    printf("%5d %5d %12.2f %12.2f %12.2f %12.2f\n", (int)p, (int)q, t_interp, t_grad, t_interp_t, t_grad_t);

    CeedVectorDestroy(&u_interp);
    CeedVectorDestroy(&v_interp);
    CeedVectorDestroy(&u_grad);
    CeedVectorDestroy(&v_grad);
    CeedVectorDestroy(&u_interp_t);
    CeedVectorDestroy(&v_interp_t);
    CeedVectorDestroy(&u_grad_t);
    CeedVectorDestroy(&v_grad_t);
    CeedBasisDestroy(&basis);
    CeedDestroy(&ceed);
  }
  return 0;
}
