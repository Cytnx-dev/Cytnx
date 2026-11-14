// Multidimensional L2 Float64 get cases. Included from Get_contiguous_double_bm.cpp
// so the original 29 rank-1 cases and their measurement path remain unchanged.
// All accessors below are in physical row-major order (identity _invmapper).
struct MultiCase {
  const char* family;
  std::vector<cytnx_uint64> shape;
  std::vector<std::vector<cytnx_uint64>> picks;  // empty => Accessor::all()
  const char* pattern;
};

std::vector<cytnx_uint64> Range(cytnx_uint64 start, cytnx_uint64 end,
                               cytnx_uint64 step) {
  std::vector<cytnx_uint64> v;
  for (auto i = start; i < end; i += step) v.push_back(i);
  return v;
}

cytnx_uint64 Product(const std::vector<cytnx_uint64>& v) {
  return std::accumulate(v.begin(), v.end(), cytnx_uint64{1},
                         std::multiplies<cytnx_uint64>{});
}

std::vector<MultiCase> MakeMultiCases() {
  std::vector<MultiCase> cases;
  const auto add_d1 = [&](cytnx_uint64 a, cytnx_uint64 b, cytnx_uint64 d) {
    cases.push_back({"D1", {a, b, d},
                     {Range(0, a, 2), {}, Range(1, d - 2, 2)}, "Strided"});
  };
  add_d1(4, 5, 7);       // T=20, U=1
  add_d1(8, 13, 11);     // T=208
  add_d1(16, 31, 19);    // T=1984
  add_d1(64, 65, 19);    // T=16640
  add_d1(128, 257, 33);  // T=246720, exact requested base
  add_d1(256, 257, 67);  // T=1052672
  add_d1(63, 127, 27);   // T=48768
  // Same sets, sizes, T/U and physical source tensor; only traversal order changes.
  for (const std::size_t i : {1u, 2u, 3u, 4u, 5u}) {
    MultiCase random = cases[i];
    random.family = "D1-R";
    random.pattern = "Random";
    std::mt19937_64 rng(1230);
    std::shuffle(random.picks[0].begin(), random.picks[0].end(), rng);
    cases.push_back(random);
  }
  const auto middle = [&](const char* family, cytnx_uint64 p,
                          cytnx_uint64 m, cytnx_uint64 u) {
    cases.push_back({family, {p, m, u},
                     {{}, Range(0, m, 2), {}}, "Strided"});
  };
  middle("D2", 8, 8, 1);        // T32, U1
  middle("D2", 32, 16, 32);    // T256, U32
  middle("D2", 128, 64, 31);   // T4096, U31
  middle("D2", 128, 64, 33);   // T4096, U33
  middle("D2", 128, 64, 64);   // T4096, U64
  middle("D2", 128, 64, 1024); // T4096, U1024
  middle("D2", 8, 8, 2048);    // T32, U2048
  middle("D2", 16, 16, 1024);  // T128, U1024
  middle("D2", 1024, 64, 33);  // T32768, U33
  middle("D2", 16, 16, 64);    // T128, U64
  middle("D2", 64, 8, 31);     // T256, U31
  // For each D3, the corresponding D2 has the same p=a*b*c, m and u.
  const auto higher = [&](cytnx_uint64 a, cytnx_uint64 b, cytnx_uint64 c,
                          cytnx_uint64 m, cytnx_uint64 u) {
    cases.push_back({"D3", {a, b, c, m, u},
                     {{}, {}, {}, Range(0, m, 2), {}}, "Strided"});
  };
  higher(2, 2, 2, 8, 1);
  higher(4, 4, 2, 16, 32);
  higher(8, 8, 2, 64, 31);
  higher(8, 8, 2, 64, 33);
  higher(8, 8, 2, 64, 64);   // exact requested D3 base
  higher(8, 8, 2, 64, 1024);
  higher(2, 2, 2, 8, 2048);
  const auto boundary = [&](const char* family, cytnx_uint64 p,
                            cytnx_uint64 m, cytnx_uint64 u,
                            cytnx_uint64 count) {
    std::vector<cytnx_uint64> indices;
    for (cytnx_uint64 k = 0; k < count; ++k) indices.push_back(1 + 3 * k);
    cases.push_back({family, {p, m, u}, {{}, indices, {}}, "Strided"});
  };
  boundary("D4a", 7, 7, 31, 2);       // T14, U31
  boundary("D4a", 17, 17, 32, 5);     // T85, U32
  boundary("D4a", 127, 65, 31, 17);   // T2159, U31
  boundary("D4a", 127, 65, 32, 17);   // T2159, U32
  boundary("D4a", 1023, 65, 32, 17);  // T17391, U32
  boundary("D4b", 5, 7, 33, 2);       // T10, U33
  boundary("D4b", 33, 17, 33, 5);     // T165, U33
  boundary("D4b", 127, 65, 33, 17);   // T2159, U33
  boundary("D4b", 513, 65, 33, 17);   // T8721, U33
  boundary("D4b", 2047, 65, 33, 17);  // T34799, U33
  return cases;
}

// Match Tensor_impl::get: strip only trailing All accessors, then match
// Storage_base::GetElem_byShape_v2: suffix strides of the REMAINING dimensions.
struct GetMetadata {
  std::vector<cytnx_uint64> offj, new_offj;
  std::vector<std::vector<cytnx_uint64>> locators;
  cytnx_uint64 total = 1, u = 1;
};
GetMetadata MakeGetMetadata(const MultiCase& c) {
  GetMetadata m;
  std::size_t rank = c.shape.size();
  while (rank && c.picks[rank - 1].empty()) {
    m.u *= c.shape[rank - 1];
    --rank;
  }
  m.locators.assign(c.picks.begin(), c.picks.begin() + rank);
  m.offj.resize(rank);
  m.new_offj.resize(rank);
  cytnx_uint64 source_stride = 1, result_stride = 1;
  for (std::size_t i = rank; i-- > 0;) {
    m.offj[i] = source_stride;
    m.new_offj[i] = result_stride;
    source_stride *= c.shape[i];
    const cytnx_uint64 n = c.picks[i].empty() ? c.shape[i] : c.picks[i].size();
    result_stride *= n;
  }
  m.total = result_stride;
  return m;
}

// Independent row-major CPU oracle. It never uses the GPU offj/new_offj
// metadata: it decodes the logical output coordinate and resolves each accessor.
cytnx_uint64 CPUPhysicalUnit(const MultiCase& c, cytnx_uint64 n,
                             cytnx_uint64 u) {
  const std::size_t rank = c.shape.size();
  cytnx_uint64 address = 0, source_stride = u;
  for (std::size_t i = rank; i-- > 0;) {
    if (i + 1 == rank && c.picks[i].empty()) continue;
    // All trailing dimensions are represented collectively by u.
    cytnx_uint64 suffix = 1;
    for (std::size_t j = i + 1; j < rank; ++j) {
      if (!c.picks[j].empty()) break;
      suffix *= c.shape[j];
    }
    if (suffix == u && c.picks[i].empty() && i + 1 < rank) continue;
    (void)source_stride;
    (void)address;
    (void)n;
    break;
  }
  // Construct the length of the non-grouped prefix directly from the shape.
  std::size_t prefix = rank;
  cytnx_uint64 trailing = 1;
  while (prefix && c.picks[prefix - 1].empty()) {
    trailing *= c.shape[--prefix];
  }
  if (trailing != u) return static_cast<cytnx_uint64>(-1);
  cytnx_uint64 stride = u, position = 0;
  for (std::size_t i = prefix; i-- > 0;) {
    const auto& pick = c.picks[i];
    const cytnx_uint64 length = pick.empty() ? c.shape[i] : pick.size();
    const cytnx_uint64 coordinate = n % length;
    n /= length;
    const cytnx_uint64 index = pick.empty() ? coordinate : pick[coordinate];
    position += index * stride;
    stride *= c.shape[i];
  }
  return position / u;
}

void BM_GetL2Multi(benchmark::State& state, const MultiCase& c, int strategy) {
  const GetMetadata m = MakeGetMetadata(c);
  const cytnx_uint64 source_count = Product(c.shape);
  const cytnx_uint64 output_count = m.total * m.u;
  if (c.shape.size() != c.picks.size() || m.locators.empty() ||
      output_count == 0 || output_count > source_count) {
    state.SkipWithError("invalid multidimensional case metadata"); return;
  }
  for (std::size_t i = 0; i < c.shape.size(); ++i)
    for (cytnx_uint64 v : c.picks[i])
      if (v >= c.shape[i]) {
        state.SkipWithError("locator exceeds source dimension"); return;
      }
  std::vector<double> host_input(source_count), expected(output_count), actual(output_count);
  for (cytnx_uint64 i = 0; i < source_count; ++i)
    host_input[i] = static_cast<double>(i) * 0.125 - 7.25;
  for (cytnx_uint64 n = 0; n < m.total; ++n) {
    const cytnx_uint64 src = CPUPhysicalUnit(c, n, m.u) * m.u;
    if (src + m.u > source_count) {
      state.SkipWithError("CPU oracle out of bounds"); return;
    }
    for (cytnx_uint64 x = 0; x < m.u; ++x)
      expected[n * m.u + x] = host_input[src + x];
  }
  double *input = nullptr, *output = nullptr;
  if (!IsOk(cudaMalloc(reinterpret_cast<void**>(&input), source_count * sizeof(double)),
            state, "cudaMalloc(input)")) return;
  if (!IsOk(cudaMalloc(reinterpret_cast<void**>(&output), output_count * sizeof(double)),
            state, "cudaMalloc(output)")) {
    cudaFree(input); return;
  }
  const auto cleanup = [&]() { cudaFree(output); cudaFree(input); };
  if (!IsOk(cudaMemcpy(input, host_input.data(), source_count * sizeof(double),
                       cudaMemcpyHostToDevice), state, "input H2D") ||
      !IsOk(cudaMemset(output, 0xa5, output_count * sizeof(double)),
            state, "output sentinel")) {
    cleanup(); return;
  }
  Invoke(strategy, output, input, m.offj, m.new_offj, m.locators, m.total, m.u);
  if (!IsOk(cudaDeviceSynchronize(), state, "correctness sync") ||
      !IsOk(cudaMemcpy(actual.data(), output, output_count * sizeof(double),
                       cudaMemcpyDeviceToHost), state, "correctness D2H")) {
    cleanup(); return;
  }
  if (std::memcmp(actual.data(), expected.data(), output_count * sizeof(double))) {
    state.SkipWithError("multidimensional correctness differs from CPU reference");
    cleanup(); return;
  }
  for (int i = 0; i < 3; ++i)
    Invoke(strategy, output, input, m.offj, m.new_offj, m.locators, m.total, m.u);
  if (!IsOk(cudaDeviceSynchronize(), state, "warm-up sync")) {
    cleanup(); return;
  }
  for (auto _ : state) {
    Invoke(strategy, output, input, m.offj, m.new_offj, m.locators, m.total, m.u);
    const auto status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
      const std::string message = std::string("timed sync: ") + cudaGetErrorString(status);
      state.SkipWithError(message.c_str());
      break;
    }
    benchmark::DoNotOptimize(output);
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(output_count) * sizeof(double));
  cleanup();
}

bool RegisterMultiCases() {
  const auto cases = MakeMultiCases();
  if (cases.size() != 40) return false;
  // D2/D3 have identical linear source trajectories at the same T and U.
  // Verify every source unit before registering *any* benchmarks.
  for (const auto& high : cases) {
    if (std::strcmp(high.family, "D3") != 0) continue;
    const GetMetadata hm = MakeGetMetadata(high);
    bool matched = false;
    for (const auto& low : cases) {
      if (std::strcmp(low.family, "D2") != 0) continue;
      const GetMetadata lm = MakeGetMetadata(low);
      if (hm.total != lm.total || hm.u != lm.u) continue;
      for (cytnx_uint64 n = 0; n < hm.total; ++n)
        if (CPUPhysicalUnit(low, n, lm.u) != CPUPhysicalUnit(high, n, hm.u))
          return false;
      matched = true;
      break;
    }
    if (!matched) return false;
  }
  for (const auto& c : cases) {
    const GetMetadata m = MakeGetMetadata(c);
    std::string shape_text;
    for (std::size_t i = 0; i < c.shape.size(); ++i) {
      if (i) shape_text += "x";
      shape_text += std::to_string(c.shape[i]);
    }
    for (int s = 0; s <= 4; ++s) {
      const auto name = std::string("GPUGetL2/") + c.family +
          "/T" + std::to_string(m.total) + "/U" + std::to_string(m.u) +
          "/" + c.pattern + "/S" + std::to_string(s) +
          "/R" + std::to_string(m.locators.size()) + "/Sh" + shape_text;
      benchmark::RegisterBenchmark(name.c_str(), [c, s](benchmark::State& state) {
        BM_GetL2Multi(state, c, s);
      })->UseRealTime()->Unit(benchmark::kMicrosecond)->Repetitions(5)->MinTime(0.5);
    }
  }
  return true;
}
const bool multi_registered = RegisterMultiCases();
