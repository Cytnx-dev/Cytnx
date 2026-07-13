# API design decisions

Recorded rulings on open API-semantics questions, with the issues they resolve
and the PRs that implement them. Contributors should treat these as settled
unless a maintainer reopens the discussion in the linked issue.

## 2026-07-06 — rulings by @yingjerkao

### 1. UniTensor elementwise arithmetic is removed from the Python surface

Resolves the #753 / #675 / #934 discussion. Elementwise `UniTensor ⊙ UniTensor`
(`+ - * /`, in-place and out-of-place) is not a meaningful tensor-network
operation — it silently ignores labels and can break block structure — so the
Python dunders now raise a guidance `TypeError` instead of being given label
semantics. Alternatives named in the error: `Contract()`, `Kron()`, scalar
arithmetic (fully retained), the Krylov solvers (`linalg.Lanczos`,
`linalg.Lanczos_Exp`, `linalg.Arnoldi`), and `ut.get_block()` Tensor
arithmetic for genuinely elementwise use. The C++ operators remain (the Krylov
solvers use them as vector-space operations) and are documented internal-only.
Implemented in PR #997.

### 2. `norm()` returns `double`; `Norm()` is deprecated

Resolves #676. New snake_case `Tensor::norm()`, `UniTensor::norm()`, and
`linalg::norm()` return a plain `double`. The old `Norm()` spellings keep
returning a rank-0 `Tensor` and are deprecated for one release (C++
`[[deprecated]]`, Python `DeprecationWarning`). Implemented in PR #1000.

### 3. Trailing underscore means in-place, everywhere

Resolves #335 / #336 / #381 (with the #421 / #422 API merges). The convention:
a trailing `_` marks an in-place method, which mutates `self` and returns it
(C++ reference / same Python object) for chaining; the non-underscore spelling
is always out-of-place. New canonical spellings added: `set_name_`,
`set_label_`, `tag_`, `convert_from_`, `combineBond_`; the old spellings are
deprecation shims for one release. **One breaking change:** bare
`combineBond()` flipped from in-place to out-of-place at the same signature —
use `combineBond_()` for the old behavior. Implemented in PR #1000.

### 4. `permute`/`reshape` accept both call syntaxes on both classes

Resolves #293. `Tensor` and `UniTensor` both accept the variadic form
(`permute(1, 2, 0)`, and variadic string labels on UniTensor) and the list
form (`permute([1, 2, 0])`). Additive, non-breaking. Implemented in PR #1000.

## Consensus positions implemented from issue threads

- **Bond is publicly immutable** (#846, position of @IvanaGyro; #841): in-place
  mutators (`set_type`, `redirect_`, `combineBond_` on `Bond`, `clear_type`,
  `group_duplicates_`, non-const `qnums()`/`syms()`) are removed in favor of
  return-new forms (`retype`, `redirect`, `combineBond`, `group_duplicates`);
  `BlockFermionicUniTensor::_signflip` is private behind a single documented
  gateway for decomposition internals. Implemented in PR #1001.
- **Shared-block safety** (#724, proposal of @manuschneider endorsed by
  @ianmccul): in-place `permute_`/`reshape_`/`contiguous_`/`combineBonds` on
  UniTensors rebuild per-block `Tensor` metadata via the non-mutating calls
  (storage stays shared, metadata detaches) instead of mutating possibly
  shared blocks. Implemented in PRs #998 and #1005.
- **Symmetry is a plain value type** (#842): the intrusive-ptr PIMPL is
  replaced by a by-value `std::variant`; the public API and the Save/Load
  byte format are preserved (pinned by binary fixtures). Two observable
  reference→value changes: `is(Symmetry, Symmetry)` now means address
  identity (copies are no longer `is` their source), and writing through the
  legacy `int &n() const` accessor no longer propagates to copies (each
  Symmetry owns its value). Implemented in PR #1010.
- **New code uses snake_case function names** in both C++ and Python (#836);
  the existing API is not mass-renamed.

## Where the detailed records live

The full execution plans (task specs, verification gates, migration tables)
for the three 2026-07 hardening phases are in [`docs/dev/plans/`](plans/).
They were written by and for agent-assisted development sessions and contain
machine-specific build paths; read them as historical decision records, not
as current build instructions.
