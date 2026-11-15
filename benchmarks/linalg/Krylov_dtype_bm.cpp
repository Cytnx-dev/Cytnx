#include <benchmark/benchmark.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "cytnx.hpp"

#ifdef UNI_GPU
  #include <cuda_runtime.h>
#endif

// Krylov solvers with Float and Double operators, on the CPU and, in CUDA builds, on the GPU.
//
// Each benchmark reports two counters:
//   * matvecs: operator applications per solver call;
//   * matvec_time: wall time per operator application.
// Lanczos "Gnd" runs a fixed budget of krylov_steps steps (a convergence criterion it cannot
// reach). Lanczos_Exp computes exp(tau H) psi with tau = -1 / ||H|| (||H|| estimated by power
// iteration), so it converges by its own criterion within the krylov_steps cap, and its step count,
// reported in matvecs, can depend on the dtype; a non-finite result is reported as an error. The
// bm_*_matvec benchmarks time the bare operator in the same dtype. Comparing matvec_time between a
// solver and its operator shows the solver's own overhead, and a Float operator whose solver runs
// at the Double operator's speed is being applied to Double vectors.
//
// The one-site GPU cases first check one application against the same operator on the CPU and
// report an error instead of a timing if they disagree. In cuQuantum builds, Network::Launch on the
// GPU returns wrong values (and can crash) for non-ComplexDouble tensors until #1245 (#1243).
//
// Operators:
//   * sandwich (Tensor path): H(psi) = L psi R for psi a d x d matrix stored as a vector of length
//     d * d, with L and R symmetric; two GEMMs per application.
//   * one-site (UniTensor path): the DMRG one-site effective Hamiltonian, contracting the left and
//     right environments and an MPO tensor with psi.
namespace krylov_dtype_bm {

  using cytnx::LinOp;
  using cytnx::Tensor;
  using cytnx::UniTensor;

  constexpr unsigned int krylov_steps = 20;
  constexpr double unreachable_cvg_crit = 1.0e-300;

  int device_for(bool on_gpu) { return on_gpu ? cytnx::Device.cuda : cytnx::Device.cpu; }

  unsigned int dtype_for(bool use_float) {
    return use_float ? cytnx::Type.Float : cytnx::Type.Double;
  }

  void sync_device(bool on_gpu) {
#ifdef UNI_GPU
    if (on_gpu) cudaDeviceSynchronize();
#endif
  }

  void report_matvecs(benchmark::State& state, std::int64_t matvecs) {
    state.counters["matvecs"] =
      benchmark::Counter(static_cast<double>(matvecs), benchmark::Counter::kAvgIterations);
    state.counters["matvec_time"] = benchmark::Counter(
      static_cast<double>(matvecs), benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
  }

  // Symmetric d x d matrix with entries in [-1, 1].
  Tensor symmetric_matrix(std::uint64_t d, unsigned int seed, unsigned int dtype, int device) {
    Tensor a = cytnx::random::uniform({d, d}, -1.0, 1.0, cytnx::Device.cpu, seed);
    Tensor s = (a + a.permute({1, 0}).contiguous()) * 0.5;
    return s.astype(dtype).to(device);
  }

  class SandwichOp : public LinOp {
   public:
    SandwichOp(std::uint64_t d, unsigned int dtype, int device)
        : LinOp("mv", d * d, dtype, device),
          dim(d),
          left(symmetric_matrix(d, 11, dtype, device)),
          right(symmetric_matrix(d, 12, dtype, device)) {}

    Tensor matvec(const Tensor& v) override {
      ++matvec_count;
      const auto d = static_cast<std::int64_t>(dim);
      return cytnx::linalg::Matmul(cytnx::linalg::Matmul(left, v.reshape({d, d})), right)
        .reshape({d * d});
    }

    Tensor start_vector() const {
      return cytnx::random::uniform({dim * dim}, -1.0, 1.0, cytnx::Device.cpu, 13)
        .astype(dtype())
        .to(device());
    }

    std::int64_t matvec_count = 0;

   private:
    std::uint64_t dim;
    Tensor left;
    Tensor right;
  };

  class OneSiteOp : public LinOp {
   public:
    OneSiteOp(std::uint64_t bond_dim, std::uint64_t phys_dim, std::uint64_t mpo_dim,
              unsigned int dtype, int device)
        : LinOp("mv", bond_dim * bond_dim * phys_dim, dtype, device) {
      // Generated on the CPU and then moved, so an operator on either device holds the same data.
      const int cpu = cytnx::Device.cpu;
      const double low = -1.0;
      const double high = 1.0;
      UniTensor l_env = UniTensor::uniform({bond_dim, mpo_dim, bond_dim}, low, high,
                                           {"vil", "Lm", "vol"}, 0, dtype, cpu);
      UniTensor r_env = UniTensor::uniform({bond_dim, mpo_dim, bond_dim}, low, high,
                                           {"vir", "Rm", "vor"}, 1, dtype, cpu);
      UniTensor mpo = UniTensor::uniform({mpo_dim, phys_dim, mpo_dim, phys_dim}, low, high,
                                         {"Lm", "pi", "Rm", "po"}, 2, dtype, cpu);
      // Make the effective Hamiltonian Hermitian.
      l_env = l_env + l_env.permute({"vol", "Lm", "vil"}).Conj().contiguous();
      r_env = r_env + r_env.permute({"vor", "Rm", "vir"}).Conj().contiguous();
      mpo = mpo + mpo.permute({"Lm", "po", "Rm", "pi"}).Conj().contiguous();
      network.FromString({"psi:'vil', 'pi', 'vir'", "L:'vil', 'Lm', 'vol'",
                          "O:'Lm', 'pi', 'Rm', 'po'", "R:'vir', 'Rm', 'vor'",
                          "TOUT:'vol'; 'po', 'vor'"});
      network.PutUniTensors({"L", "O", "R"}, {l_env.to(device), mpo.to(device), r_env.to(device)});
      start = UniTensor::uniform({bond_dim, phys_dim, bond_dim}, low, high, {"vil", "pi", "vir"}, 3,
                                 dtype, cpu)
                .to(device);
    }

    UniTensor matvec(const UniTensor& psi) override {
      ++matvec_count;
      network.PutUniTensor("psi", psi);
      return network.Launch().relabel_(psi.labels());
    }

    const UniTensor& start_vector() const { return start; }

    std::int64_t matvec_count = 0;

   private:
    cytnx::Network network;
    UniTensor start;
  };

  constexpr std::uint64_t one_site_phys_dim = 2;
  constexpr std::uint64_t one_site_mpo_dim = 5;

  // Euclidean norm, as a double, for either dtype on either device.
  double norm_of(const UniTensor& u) {
    return u.Norm().astype(cytnx::Type.Double).to(cytnx::Device.cpu).item<double>();
  }

  // Checks one application of a GPU one-site operator against the same operator on the CPU. If
  // they disagree, as they do when the GPU contraction path is broken (#1243), marks the benchmark
  // as failed and returns false.
  bool one_site_gpu_matches_cpu(benchmark::State& state, OneSiteOp& gpu_op, std::uint64_t bond_dim,
                                bool use_float) {
    OneSiteOp cpu_op(bond_dim, one_site_phys_dim, one_site_mpo_dim, dtype_for(use_float),
                     cytnx::Device.cpu);
    const UniTensor expected = cpu_op.matvec(cpu_op.start_vector());
    const UniTensor actual = gpu_op.matvec(gpu_op.start_vector()).to(cytnx::Device.cpu);
    const double difference = norm_of(expected - actual) / norm_of(expected);
    const double tolerance = use_float ? 1.0e-4 : 1.0e-10;
    if (difference <= tolerance) return true;  // also false for NaN
    state.SkipWithError(("GPU one-site operator disagrees with the CPU: relative difference " +
                         std::to_string(difference))
                          .c_str());
    return false;
  }

  // Estimate of ||H|| (its largest eigenvalue in magnitude) by power iteration.
  double operator_norm_estimate(OneSiteOp& op) {
    UniTensor v = op.start_vector() / norm_of(op.start_vector());
    double estimate = 0.0;
    for (int iteration = 0; iteration < 10; ++iteration) {
      const UniTensor w = op.matvec(v);
      estimate = norm_of(w);
      v = w / estimate;
    }
    return estimate;
  }

  void bm_sandwich_matvec(benchmark::State& state, bool on_gpu, bool use_float) {
    SandwichOp op(static_cast<std::uint64_t>(state.range(0)), dtype_for(use_float),
                  device_for(on_gpu));
    const Tensor v = op.start_vector();
    op.matvec(v);
    sync_device(on_gpu);
    op.matvec_count = 0;
    for ([[maybe_unused]] auto iteration : state) {
      Tensor w = op.matvec(v);
      sync_device(on_gpu);
      benchmark::DoNotOptimize(w);
    }
    report_matvecs(state, op.matvec_count);
  }

  void bm_lanczos_gnd_tensor(benchmark::State& state, bool on_gpu, bool use_float) {
    SandwichOp op(static_cast<std::uint64_t>(state.range(0)), dtype_for(use_float),
                  device_for(on_gpu));
    const Tensor v0 = op.start_vector();
    const std::uint64_t k = 1;
    const std::uint32_t max_krydim = 0;
    const auto run = [&]() {
      return cytnx::linalg::Lanczos(&op, v0, "Gnd", unreachable_cvg_crit, krylov_steps, k,
                                    /*is_V=*/true, /*is_row=*/false, max_krydim);
    };
    run();
    sync_device(on_gpu);
    op.matvec_count = 0;
    for ([[maybe_unused]] auto iteration : state) {
      auto result = run();
      sync_device(on_gpu);
      benchmark::DoNotOptimize(result);
    }
    report_matvecs(state, op.matvec_count);
  }

  void bm_one_site_matvec(benchmark::State& state, bool on_gpu, bool use_float) {
    const auto bond_dim = static_cast<std::uint64_t>(state.range(0));
    OneSiteOp op(bond_dim, one_site_phys_dim, one_site_mpo_dim, dtype_for(use_float),
                 device_for(on_gpu));
    if (on_gpu && !one_site_gpu_matches_cpu(state, op, bond_dim, use_float)) return;
    op.matvec(op.start_vector());
    sync_device(on_gpu);
    op.matvec_count = 0;
    for ([[maybe_unused]] auto iteration : state) {
      UniTensor w = op.matvec(op.start_vector());
      sync_device(on_gpu);
      benchmark::DoNotOptimize(w);
    }
    report_matvecs(state, op.matvec_count);
  }

  void bm_lanczos_gnd_unitensor(benchmark::State& state, bool on_gpu, bool use_float) {
    const auto bond_dim = static_cast<std::uint64_t>(state.range(0));
    OneSiteOp op(bond_dim, one_site_phys_dim, one_site_mpo_dim, dtype_for(use_float),
                 device_for(on_gpu));
    if (on_gpu && !one_site_gpu_matches_cpu(state, op, bond_dim, use_float)) return;
    const std::uint64_t k = 1;
    const std::uint32_t max_krydim = 0;
    const auto run = [&]() {
      return cytnx::linalg::Lanczos(&op, op.start_vector(), "Gnd", unreachable_cvg_crit,
                                    krylov_steps, k, /*is_V=*/true, /*is_row=*/false, max_krydim);
    };
    run();
    sync_device(on_gpu);
    op.matvec_count = 0;
    for ([[maybe_unused]] auto iteration : state) {
      auto result = run();
      sync_device(on_gpu);
      benchmark::DoNotOptimize(result);
    }
    report_matvecs(state, op.matvec_count);
  }

  void bm_lanczos_exp_unitensor(benchmark::State& state, bool on_gpu, bool use_float) {
    OneSiteOp op(static_cast<std::uint64_t>(state.range(0)), one_site_phys_dim, one_site_mpo_dim,
                 dtype_for(use_float), device_for(on_gpu));
    // |tau| * ||H|| = 1: a time step the expansion resolves within krylov_steps. A fixed tau would
    // overflow exp(tau H) for these operators, whose spectra grow with the bond dimension.
    const double tau = -1.0 / operator_norm_estimate(op);
    const double cvg_crit = 1.0e-8;
    const auto run = [&]() {
      return cytnx::linalg::Lanczos_Exp(&op, op.start_vector(), tau, cvg_crit, krylov_steps);
    };
    if (!std::isfinite(norm_of(run()))) {
      state.SkipWithError("Lanczos_Exp returned a non-finite result");
      return;
    }
    sync_device(on_gpu);
    op.matvec_count = 0;
    for ([[maybe_unused]] auto iteration : state) {
      UniTensor result = run();
      sync_device(on_gpu);
      benchmark::DoNotOptimize(result);
    }
    report_matvecs(state, op.matvec_count);
  }

  // The same sizes on the CPU and the GPU, so the two can be compared directly; on one CPU thread
  // the largest sizes take several seconds per solver call. Real time, not CPU time: matvec_time is
  // a rate over the benchmark's time, and with a threaded BLAS or a GPU the main thread's CPU time
  // is not the elapsed time.
  BENCHMARK_CAPTURE(bm_sandwich_matvec, cpu_float, false, true)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
  BENCHMARK_CAPTURE(bm_sandwich_matvec, cpu_double, false, false)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
  BENCHMARK_CAPTURE(bm_lanczos_gnd_tensor, cpu_float, false, true)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
  BENCHMARK_CAPTURE(bm_lanczos_gnd_tensor, cpu_double, false, false)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
  BENCHMARK_CAPTURE(bm_one_site_matvec, cpu_float, false, true)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
  BENCHMARK_CAPTURE(bm_one_site_matvec, cpu_double, false, false)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
  BENCHMARK_CAPTURE(bm_lanczos_gnd_unitensor, cpu_float, false, true)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
  BENCHMARK_CAPTURE(bm_lanczos_gnd_unitensor, cpu_double, false, false)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
  BENCHMARK_CAPTURE(bm_lanczos_exp_unitensor, cpu_float, false, true)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
  BENCHMARK_CAPTURE(bm_lanczos_exp_unitensor, cpu_double, false, false)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

#ifdef UNI_GPU
  // Lanczos_Exp has no GPU registration: it rejects CUDA devices ("Lanczos_Exp still does not
  // support cuda devices").
  BENCHMARK_CAPTURE(bm_sandwich_matvec, gpu_float, true, true)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
  BENCHMARK_CAPTURE(bm_sandwich_matvec, gpu_double, true, false)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
  BENCHMARK_CAPTURE(bm_lanczos_gnd_tensor, gpu_float, true, true)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
  BENCHMARK_CAPTURE(bm_lanczos_gnd_tensor, gpu_double, true, false)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
  BENCHMARK_CAPTURE(bm_one_site_matvec, gpu_float, true, true)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
  BENCHMARK_CAPTURE(bm_one_site_matvec, gpu_double, true, false)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
  BENCHMARK_CAPTURE(bm_lanczos_gnd_unitensor, gpu_float, true, true)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
  BENCHMARK_CAPTURE(bm_lanczos_gnd_unitensor, gpu_double, true, false)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
#endif  // UNI_GPU

}  // namespace krylov_dtype_bm
