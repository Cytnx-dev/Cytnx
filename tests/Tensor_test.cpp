#include "Tensor_test.h"

#include <filesystem>
#include <fstream>

#include "test_tools.h"

namespace cytnx {
  namespace test {
    namespace {

      TEST_F(TensorTest, Constructor) {
        Tensor A;
        EXPECT_EQ(A.dtype(), Type.Void);
        EXPECT_TRUE(A.is_void());
        EXPECT_FALSE(A.is_scalar());
        EXPECT_EQ(A.device(), Device.cpu);
        EXPECT_EQ(A.shape().size(), 0);
        EXPECT_EQ(A.is_contiguous(), true);

        Tensor B({3, 4, 5});
        EXPECT_EQ(B.dtype(), Type.Double);
        EXPECT_FALSE(B.is_void());
        EXPECT_EQ(B.device(), Device.cpu);
        EXPECT_EQ(B.shape().size(), 3);
        EXPECT_EQ(B.shape()[0], 3);
        EXPECT_EQ(B.shape()[1], 4);
        EXPECT_EQ(B.shape()[2], 5);
        EXPECT_EQ(B.is_contiguous(), true);

        Tensor C({3, 4, 5}, Type.Double);
        EXPECT_EQ(C.dtype(), Type.Double);
        EXPECT_FALSE(C.is_void());
        EXPECT_EQ(C.device(), Device.cpu);
        EXPECT_EQ(C.shape().size(), 3);
        EXPECT_EQ(C.shape()[0], 3);
        EXPECT_EQ(C.shape()[1], 4);
        EXPECT_EQ(C.shape()[2], 5);
        EXPECT_EQ(C.is_contiguous(), true);

        Tensor S({}, Type.Double);
        EXPECT_EQ(S.dtype(), Type.Double);
        EXPECT_FALSE(S.is_void());
        EXPECT_EQ(S.device(), Device.cpu);
        EXPECT_EQ(S.shape().size(), 0);
        EXPECT_EQ(S.rank(), 0);
        EXPECT_TRUE(S.is_scalar());
        EXPECT_EQ(S.storage().size(), 1);
        EXPECT_EQ(S.is_contiguous(), true);

        Tensor T({2, 0, 3}, Type.Float);
        EXPECT_EQ(T.shape(), (std::vector<cytnx_uint64>{2, 0, 3}));
        EXPECT_EQ(T.rank(), 3);
        EXPECT_EQ(T.size(), 0);
        EXPECT_TRUE(T.is_empty());
        EXPECT_FALSE(T.is_void());
        EXPECT_FALSE(T.is_scalar());
#ifdef UNI_GPU
        Tensor D({3, 4, 5}, Type.Double, Device.cuda);
        EXPECT_EQ(D.dtype(), Type.Double);
        EXPECT_EQ(D.device(), Device.cuda);
        EXPECT_EQ(D.shape().size(), 3);
        EXPECT_EQ(D.shape()[0], 3);
        EXPECT_EQ(D.shape()[1], 4);
        EXPECT_EQ(D.shape()[2], 5);
        EXPECT_EQ(D.is_contiguous(), true);

        Tensor E({3, 4, 5}, Type.Double, Device.cuda, true);
        EXPECT_EQ(E.dtype(), Type.Double);
        EXPECT_EQ(E.device(), Device.cuda);
        EXPECT_EQ(E.shape().size(), 3);
        EXPECT_EQ(E.shape()[0], 3);
        EXPECT_EQ(E.shape()[1], 4);
        EXPECT_EQ(E.shape()[2], 5);
        EXPECT_EQ(E.is_contiguous(), true);

        Tensor F({3, 4, 5}, Type.Double, Device.cuda, false);
        EXPECT_EQ(F.dtype(), Type.Double);
        EXPECT_EQ(F.device(), Device.cuda);
        EXPECT_EQ(F.shape().size(), 3);
        EXPECT_EQ(F.shape()[0], 3);
        EXPECT_EQ(F.shape()[1], 4);
        EXPECT_EQ(F.shape()[2], 5);
        EXPECT_EQ(F.is_contiguous(), true);
#endif
      }

      TEST_F(TensorTest, ShapeGeneratorsDistinguishScalarAndRankOne) {
        Tensor zero_scalar = zeros({});
        EXPECT_EQ(zero_scalar.shape().size(), 0);
        EXPECT_TRUE(zero_scalar.is_scalar());
        EXPECT_DOUBLE_EQ(zero_scalar.item<double>(), 0.0);

        Tensor one_scalar = ones({}, Type.Float);
        EXPECT_EQ(one_scalar.shape().size(), 0);
        EXPECT_TRUE(one_scalar.is_scalar());
        EXPECT_FLOAT_EQ(one_scalar.item<cytnx_float>(), 1.0f);

        Tensor zero_vector = zeros({5});
        EXPECT_EQ(zero_vector.shape(), (std::vector<cytnx_uint64>{5}));
        EXPECT_FALSE(zero_vector.is_scalar());

        Tensor one_vector = ones({5}, Type.Int64);
        EXPECT_EQ(one_vector.shape(), (std::vector<cytnx_uint64>{5}));
        EXPECT_FALSE(one_vector.is_scalar());
        EXPECT_EQ(one_vector.at<cytnx_int64>({0}), 1);

        Tensor empty_vector = ones({0}, Type.Double);
        EXPECT_EQ(empty_vector.shape(), (std::vector<cytnx_uint64>{0}));
        EXPECT_EQ(empty_vector.size(), 0);
        EXPECT_TRUE(empty_vector.is_empty());
        EXPECT_THROW(empty_vector.item<double>(), std::logic_error);
        EXPECT_THROW(empty_vector.at<double>({0}), std::logic_error);
      }

      TEST_F(TensorTest, VoidTensorAtEmptyLocatorThrows) {
        Tensor uninitialized;
        EXPECT_EQ(uninitialized.dtype(), Type.Void);
        EXPECT_TRUE(uninitialized.is_void());
        EXPECT_THROW(uninitialized.at<double>({}), std::logic_error);
        EXPECT_THROW(uninitialized.at({}), std::logic_error);

        const Tensor &const_uninitialized = uninitialized;
        EXPECT_THROW(const_uninitialized.at({}), std::logic_error);
      }

      TEST_F(TensorTest, CopyConstructor) {
#ifdef UNI_GPU
        Tensor A({3, 4, 5}, Type.Double, Device.cuda, false);
        Tensor B(A);
        EXPECT_EQ(B.dtype(), Type.Double);
        EXPECT_EQ(B.device(), Device.cuda);
        EXPECT_EQ(B.shape().size(), 3);
        EXPECT_EQ(B.shape()[0], 3);
        EXPECT_EQ(B.shape()[1], 4);
        EXPECT_EQ(B.shape()[2], 5);
        EXPECT_EQ(B.is_contiguous(), true);

        Tensor C = A;
        EXPECT_EQ(C.dtype(), Type.Double);
        EXPECT_EQ(C.device(), Device.cuda);
        EXPECT_EQ(C.shape().size(), 3);
        EXPECT_EQ(C.shape()[0], 3);
        EXPECT_EQ(C.shape()[1], 4);
        EXPECT_EQ(C.shape()[2], 5);
        EXPECT_EQ(C.is_contiguous(), true);
#endif

        Tensor D;
        D = tarcomplex3456;
        EXPECT_EQ(D.dtype(), Type.ComplexDouble);
        EXPECT_EQ(D.device(), Device.cpu);
        EXPECT_EQ(D.shape().size(), 4);
        EXPECT_EQ(D.shape()[0], 3);
        EXPECT_EQ(D.shape()[1], 4);
        EXPECT_EQ(D.shape()[2], 5);
        EXPECT_EQ(D.shape()[3], 6);
        EXPECT_EQ(D.is_contiguous(), true);

        Tensor E;
        E = tarcomplex3456.permute({1, 2, 3, 0});
        EXPECT_EQ(E.dtype(), Type.ComplexDouble);
        EXPECT_EQ(E.device(), Device.cpu);
        EXPECT_EQ(E.shape().size(), 4);
        EXPECT_EQ(E.shape()[0], 4);
        EXPECT_EQ(E.shape()[1], 5);
        EXPECT_EQ(E.shape()[2], 6);
        EXPECT_EQ(E.shape()[3], 3);
        EXPECT_EQ(E.is_contiguous(), false);
      }

      TEST_F(TensorTest, Shape) {
        Tensor A({3, 4, 5}, Type.Double, Device.cpu, false);
        EXPECT_EQ(A.shape().size(), 3);
        EXPECT_EQ(A.shape()[0], 3);
        EXPECT_EQ(A.shape()[1], 4);
        EXPECT_EQ(A.shape()[2], 5);
        EXPECT_EQ(A.is_contiguous(), true);

        A.reshape_({4, 5, 3});
        EXPECT_EQ(A.shape().size(), 3);
        EXPECT_EQ(A.shape()[0], 4);
        EXPECT_EQ(A.shape()[1], 5);
        EXPECT_EQ(A.shape()[2], 3);

        A.reshape_(3, 4, 5);
        EXPECT_EQ(A.shape().size(), 3);
        EXPECT_EQ(A.shape()[0], 3);
        EXPECT_EQ(A.shape()[1], 4);
        EXPECT_EQ(A.shape()[2], 5);

        auto tmp = A.reshape({4, 3, 5});
        EXPECT_EQ(tmp.shape().size(), 3);
        EXPECT_EQ(tmp.shape()[0], 4);
        EXPECT_EQ(tmp.shape()[1], 3);
        EXPECT_EQ(tmp.shape()[2], 5);

        tmp = A.reshape(4, 5, 3);
        EXPECT_EQ(tmp.shape().size(), 3);
        EXPECT_EQ(tmp.shape()[0], 4);
        EXPECT_EQ(tmp.shape()[1], 5);
        EXPECT_EQ(tmp.shape()[2], 3);

        Tensor B({1}, Type.Double, Device.cpu, true);
        EXPECT_EQ(B.shape().size(), 1);
        EXPECT_EQ(B.shape()[0], 1);

        B.reshape_({1, 1});
        EXPECT_EQ(B.shape().size(), 2);
        EXPECT_EQ(B.shape()[0], 1);
        EXPECT_EQ(B.shape()[1], 1);

        B.reshape_({1});
        EXPECT_EQ(B.shape().size(), 1);
        EXPECT_EQ(B.shape()[0], 1);

        B.reshape_(1, 1, 1);
        EXPECT_EQ(B.shape().size(), 3);
        EXPECT_EQ(B.shape()[0], 1);
        EXPECT_EQ(B.shape()[1], 1);
        EXPECT_EQ(B.shape()[2], 1);

        B.reshape_(std::vector<cytnx_int64>{});
        EXPECT_EQ(B.shape().size(), 0);
        B.reshape_(std::vector<cytnx_uint64>{1});
        EXPECT_EQ(B.shape().size(), 1);
        EXPECT_EQ(B.shape()[0], 1);
        EXPECT_FALSE(B.is_scalar());

        Tensor empty({0}, Type.Double, Device.cpu, true);
        EXPECT_TRUE(empty.is_empty());
        EXPECT_EQ(empty.reshape({2, 0, 3}).shape(), (std::vector<cytnx_uint64>{2, 0, 3}));
        EXPECT_EQ(empty.reshape({-1}).shape(), (std::vector<cytnx_uint64>{0}));
        EXPECT_THROW(empty.reshape({0, -1}), std::logic_error);
      }

      TEST_F(TensorTest, ZeroExtentArithmeticAndSlicing) {
        Tensor empty = zeros({2, 0, 3}, Type.Float);
        Tensor other = ones({2, 0, 3}, Type.Double);
        Tensor scalar = ones({}, Type.Double);

        Tensor sum = empty + other;
        EXPECT_EQ(sum.shape(), empty.shape());
        EXPECT_EQ(sum.dtype(), Type.Double);
        EXPECT_TRUE(sum.is_empty());

        Tensor broadcast = empty + scalar;
        EXPECT_EQ(broadcast.shape(), empty.shape());
        EXPECT_EQ(broadcast.dtype(), Type.Double);
        EXPECT_TRUE(broadcast.is_empty());

        EXPECT_TRUE((empty - other).is_empty());
        EXPECT_TRUE((empty * other).is_empty());
        EXPECT_TRUE((empty / other).is_empty());
        EXPECT_TRUE((empty + 2.0).is_empty());
        EXPECT_TRUE((empty - 2.0).is_empty());
        EXPECT_TRUE((empty * 2.0).is_empty());
        EXPECT_TRUE((empty / 2.0).is_empty());
        EXPECT_TRUE((2.0 + empty).is_empty());
        EXPECT_TRUE((2.0 - empty).is_empty());
        EXPECT_TRUE((2.0 * empty).is_empty());
        EXPECT_TRUE((2.0 / empty).is_empty());

        // `scalar` is a genuine rank-0 Double tensor, not a python weak scalar, so an
        // in-place op promotes the Float lhs to Double (#941) -- matching the
        // out-of-place `empty + scalar` above and the non-empty path. A zero-extent
        // lhs still takes this dtype promotion even though there is nothing to compute.
        empty += scalar;
        EXPECT_EQ(empty.shape(), (std::vector<cytnx_uint64>{2, 0, 3}));
        EXPECT_EQ(empty.dtype(), Type.Double);
        EXPECT_TRUE(empty.is_empty());

        Tensor slice = empty.get({Accessor::all(), Accessor::all(), Accessor::all()});
        EXPECT_EQ(slice.shape(), empty.shape());
        EXPECT_TRUE(slice.is_empty());

        EXPECT_NO_THROW(empty.set({Accessor::all(), Accessor::all(), Accessor::all()}, other));
        EXPECT_THROW(empty.set({Accessor::all(), Accessor::all(), Accessor::all()},
                               Tensor({0, 2}, Type.Double)),
                     std::logic_error);
      }

      TEST_F(TensorTest, VoidInplaceArithmeticThrows) {
        Tensor scalar = ones({}, Type.Double);
        Tensor uninitialized;

        EXPECT_THROW(linalg::iAdd(scalar, uninitialized), std::logic_error);
        EXPECT_THROW(linalg::iSub(scalar, uninitialized), std::logic_error);
        EXPECT_THROW(linalg::iMul(scalar, uninitialized), std::logic_error);
        EXPECT_THROW(linalg::iDiv(scalar, uninitialized), std::logic_error);
      }

      TEST_F(TensorTest, Permute) {
        Tensor A({3, 4, 5}, Type.Double, Device.cpu);
        EXPECT_EQ(A.shape().size(), 3);
        EXPECT_EQ(A.shape()[0], 3);
        EXPECT_EQ(A.shape()[1], 4);
        EXPECT_EQ(A.shape()[2], 5);
        EXPECT_EQ(A.is_contiguous(), true);

        A.permute_({2, 1, 0});
        EXPECT_EQ(A.shape().size(), 3);
        EXPECT_EQ(A.shape()[0], 5);
        EXPECT_EQ(A.shape()[1], 4);
        EXPECT_EQ(A.shape()[2], 3);
        EXPECT_EQ(A.is_contiguous(), false);

        A.permute_(2, 1, 0);
        EXPECT_EQ(A.shape().size(), 3);
        EXPECT_EQ(A.shape()[0], 3);
        EXPECT_EQ(A.shape()[1], 4);
        EXPECT_EQ(A.shape()[2], 5);
        EXPECT_EQ(A.is_contiguous(), true);

        auto tmp = A.permute({2, 0, 1});
        EXPECT_EQ(tmp.shape().size(), 3);
        EXPECT_EQ(tmp.shape()[0], 5);
        EXPECT_EQ(tmp.shape()[1], 3);
        EXPECT_EQ(tmp.shape()[2], 4);
        EXPECT_EQ(tmp.is_contiguous(), false);

        tmp = A.permute(2, 0, 1);
        EXPECT_EQ(tmp.shape().size(), 3);
        EXPECT_EQ(tmp.shape()[0], 5);
        EXPECT_EQ(tmp.shape()[1], 3);
        EXPECT_EQ(tmp.shape()[2], 4);
        EXPECT_EQ(tmp.is_contiguous(), false);

        Tensor B({1}, Type.Double, Device.cpu, true);
        EXPECT_EQ(B.shape().size(), 1);
        EXPECT_EQ(B.shape()[0], 1);

        B.permute_({0});
        EXPECT_EQ(B.shape().size(), 1);
        EXPECT_EQ(B.shape()[0], 1);

        B.permute_(0);
        EXPECT_EQ(B.shape().size(), 1);
        EXPECT_EQ(B.shape()[0], 1);

        Tensor empty({0}, Type.Double, Device.cpu, true);
        empty.permute_({0});
        EXPECT_TRUE(empty.is_empty());
      }

      TEST_F(TensorTest, Get) {
        Tensor tmp = tzero3456(":", ":", ":", ":");
        EXPECT_EQ(tmp.shape().size(), 4);
        EXPECT_EQ(tmp.shape()[0], 3);
        EXPECT_EQ(tmp.shape()[1], 4);
        EXPECT_EQ(tmp.shape()[2], 5);
        EXPECT_EQ(tmp.shape()[3], 6);
        EXPECT_EQ(tmp.is_contiguous(), true);

        tmp = tzero3456(0, ":", ":", ":");
        EXPECT_EQ(tmp.shape().size(), 3);
        EXPECT_EQ(tmp.shape()[0], 4);
        EXPECT_EQ(tmp.shape()[1], 5);
        EXPECT_EQ(tmp.shape()[2], 6);
        EXPECT_EQ(tmp.is_contiguous(), true);

        tmp = tzero3456(0, 0, ":", ":");
        EXPECT_EQ(tmp.shape().size(), 2);
        EXPECT_EQ(tmp.shape()[0], 5);
        EXPECT_EQ(tmp.shape()[1], 6);
        EXPECT_EQ(tmp.is_contiguous(), true);

        tmp = tzero3456(1, 0, 0, ":");
        EXPECT_EQ(tmp.shape().size(), 1);
        EXPECT_EQ(tmp.shape()[0], 6);
        EXPECT_EQ(tmp.is_contiguous(), true);

        tmp = tzero3456(0, 1, 4, 4);
        EXPECT_EQ(tmp.dtype(), Type.ComplexDouble);
        EXPECT_TRUE(tmp.is_scalar());
        EXPECT_EQ(tmp.storage().size(), 1);
        EXPECT_EQ(tmp.item<cytnx_complex128>(), cytnx_complex128(0, 0));
        EXPECT_EQ(tmp.is_contiguous(), true);

        tmp =
          tarcomplex3456.get({Accessor::all(), Accessor::all(), Accessor::all(), Accessor::all()});
        EXPECT_EQ(tmp.shape().size(), 4);
        EXPECT_EQ(tmp.shape()[0], 3);
        EXPECT_EQ(tmp.shape()[1], 4);
        EXPECT_EQ(tmp.shape()[2], 5);
        EXPECT_EQ(tmp.shape()[3], 6);
        EXPECT_EQ(tmp.is_contiguous(), true);

        tmp = tarcomplex3456(":1", ":", ":", ":");
        EXPECT_EQ(tmp.shape().size(), 4);
        EXPECT_EQ(tmp.shape()[0], 1);
        EXPECT_EQ(tmp.shape()[1], 4);
        EXPECT_EQ(tmp.shape()[2], 5);
        EXPECT_EQ(tmp.shape()[3], 6);
        EXPECT_EQ(tmp.is_contiguous(), true);

        tmp = tarcomplex3456("0:1", ":", ":", ":");
        EXPECT_EQ(tmp.shape().size(), 4);
        EXPECT_EQ(tmp.shape()[0], 1);
        EXPECT_EQ(tmp.shape()[1], 4);
        EXPECT_EQ(tmp.shape()[2], 5);
        EXPECT_EQ(tmp.shape()[3], 6);
        EXPECT_EQ(tmp.is_contiguous(), true);

        tmp = tarcomplex3456("0:2", ":", ":", ":");
        EXPECT_EQ(tmp.shape().size(), 4);
        EXPECT_EQ(tmp.shape()[0], 2);
        EXPECT_EQ(tmp.shape()[1], 4);
        EXPECT_EQ(tmp.shape()[2], 5);
        EXPECT_EQ(tmp.shape()[3], 6);

        tmp = tarcomplex3456("1:2", ":", ":", ":");
        EXPECT_EQ(tmp.shape().size(), 4);
        EXPECT_EQ(tmp.shape()[0], 1);
        EXPECT_EQ(tmp.shape()[1], 4);
        EXPECT_EQ(tmp.shape()[2], 5);
        EXPECT_EQ(tmp.shape()[3], 6);
        test::AreNearlyEqTensor(tmp, tslice1);
      }

      TEST_F(TensorTest, Set) {
        auto tmp = tar3456.clone();
        tar3456(1, 2, 3, 4) = -1999;
        for (size_t i = 0; i < 3; i++)
          for (size_t j = 0; j < 4; j++)
            for (size_t k = 0; k < 5; k++)
              for (size_t l = 0; l < 6; l++)
                if (i == 1 && j == 2 && k == 3 && l == 4) {
                  EXPECT_EQ(tar3456(i, j, k, l).item().real(), -1999);
                } else {
                  EXPECT_EQ(tar3456(i, j, k, l).item().real(), tmp(i, j, k, l).item().real());
                }
      }

      TEST_F(TensorTest, RankZeroScalarAccessAndSet) {
        Tensor scalar({}, Type.Double);
        scalar.item<double>() = 3.25;
        EXPECT_DOUBLE_EQ(scalar.item<double>(), 3.25);
        EXPECT_DOUBLE_EQ(scalar.at<double>({}), 3.25);

        scalar.at<double>({}) = 4.5;
        EXPECT_DOUBLE_EQ(scalar.item<double>(), 4.5);

        Tensor selected = scalar.get(std::vector<Accessor>{});
        EXPECT_EQ(selected.shape().size(), 0);
        EXPECT_TRUE(selected.is_scalar());
        EXPECT_DOUBLE_EQ(selected.item<double>(), 4.5);

        Tensor replacement({}, Type.Double);
        replacement.item<double>() = -2.0;
        scalar.set(std::vector<Accessor>{}, replacement);
        EXPECT_DOUBLE_EQ(scalar.item<double>(), -2.0);

        scalar.set(std::vector<Accessor>{}, 7.0);
        EXPECT_DOUBLE_EQ(scalar.item<double>(), 7.0);

        Tensor vector = zeros({3}, Type.Double);
        Tensor shape_one({1}, Type.Double);
        shape_one.at<double>({0}) = 9.0;
        scalar.set(std::vector<Accessor>{}, shape_one);
        EXPECT_DOUBLE_EQ(scalar.item<double>(), 9.0);

        vector.set(std::vector<Accessor>{}, shape_one);
        for (int i = 0; i < 3; ++i) EXPECT_DOUBLE_EQ(vector.at<double>({i}), 9.0);

        Tensor shape_one_one({1, 1}, Type.Double);
        shape_one_one.at<double>({0, 0}) = 6.0;
        vector.set(std::vector<Accessor>{Accessor(0)}, shape_one_one);
        EXPECT_DOUBLE_EQ(vector.at<double>({0}), 6.0);

        Tensor scalar_rhs({}, Type.Double);
        scalar_rhs.item<double>() = 8.0;
        vector.set(std::vector<Accessor>{Accessor(0)}, scalar_rhs);
        EXPECT_DOUBLE_EQ(vector.at<double>({0}), 8.0);

        Tensor scalar_plus_shape_one = scalar_rhs + shape_one;
        EXPECT_EQ(scalar_plus_shape_one.shape(), (std::vector<cytnx_uint64>{1}));
        EXPECT_FALSE(scalar_plus_shape_one.is_scalar());
        EXPECT_DOUBLE_EQ(scalar_plus_shape_one.at<double>({0}), 17.0);

        Tensor scalar_plus_scalar = scalar + scalar_rhs;
        EXPECT_TRUE(scalar_plus_scalar.is_scalar());
        EXPECT_DOUBLE_EQ(scalar_plus_scalar.item<double>(), 17.0);
      }

      TEST_F(TensorTest, RankZeroBroadcastArithmetic) {
        Tensor scalar({}, Type.Double);
        scalar.item<double>() = 2.0;
        Tensor vec = arange(0, 3, 1, Type.Double);

        Tensor out = scalar + vec;
        EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
        EXPECT_DOUBLE_EQ(out.at<double>({0}), 2.0);
        EXPECT_DOUBLE_EQ(out.at<double>({1}), 3.0);
        EXPECT_DOUBLE_EQ(out.at<double>({2}), 4.0);

        out = vec + scalar;
        EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
        EXPECT_DOUBLE_EQ(out.at<double>({0}), 2.0);
        EXPECT_DOUBLE_EQ(out.at<double>({1}), 3.0);
        EXPECT_DOUBLE_EQ(out.at<double>({2}), 4.0);

        out = scalar - vec;
        EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
        EXPECT_DOUBLE_EQ(out.at<double>({0}), 2.0);
        EXPECT_DOUBLE_EQ(out.at<double>({1}), 1.0);
        EXPECT_DOUBLE_EQ(out.at<double>({2}), 0.0);

        out = vec - scalar;
        EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
        EXPECT_DOUBLE_EQ(out.at<double>({0}), -2.0);
        EXPECT_DOUBLE_EQ(out.at<double>({1}), -1.0);
        EXPECT_DOUBLE_EQ(out.at<double>({2}), 0.0);

        Tensor scalar2({}, Type.Double);
        scalar2.item<double>() = 5.0;
        out = scalar * scalar2;
        EXPECT_EQ(out.shape().size(), 0);
        EXPECT_DOUBLE_EQ(out.item<double>(), 10.0);

        out = scalar * vec;
        EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
        EXPECT_DOUBLE_EQ(out.at<double>({0}), 0.0);
        EXPECT_DOUBLE_EQ(out.at<double>({1}), 2.0);
        EXPECT_DOUBLE_EQ(out.at<double>({2}), 4.0);

        out = vec * scalar;
        EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
        EXPECT_DOUBLE_EQ(out.at<double>({0}), 0.0);
        EXPECT_DOUBLE_EQ(out.at<double>({1}), 2.0);
        EXPECT_DOUBLE_EQ(out.at<double>({2}), 4.0);

        Tensor denom = arange(1, 4, 1, Type.Double);
        out = scalar / denom;
        EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
        EXPECT_DOUBLE_EQ(out.at<double>({0}), 2.0);
        EXPECT_DOUBLE_EQ(out.at<double>({1}), 1.0);
        EXPECT_DOUBLE_EQ(out.at<double>({2}), 2.0 / 3.0);

        out = denom / scalar;
        EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
        EXPECT_DOUBLE_EQ(out.at<double>({0}), 0.5);
        EXPECT_DOUBLE_EQ(out.at<double>({1}), 1.0);
        EXPECT_DOUBLE_EQ(out.at<double>({2}), 1.5);

        Tensor mod_scalar({}, Type.Int64);
        mod_scalar.item<cytnx_int64>() = 5;
        Tensor mod_vec = arange(2, 5, 1, Type.Int64);

        out = linalg::Mod(mod_scalar, mod_vec);
        EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
        EXPECT_EQ(out.at<cytnx_int64>({0}), 1);
        EXPECT_EQ(out.at<cytnx_int64>({1}), 2);
        EXPECT_EQ(out.at<cytnx_int64>({2}), 1);

        out = linalg::Mod(mod_vec, mod_scalar);
        EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
        EXPECT_EQ(out.at<cytnx_int64>({0}), 2);
        EXPECT_EQ(out.at<cytnx_int64>({1}), 3);
        EXPECT_EQ(out.at<cytnx_int64>({2}), 4);

        vec += scalar;
        EXPECT_EQ(vec.shape(), (std::vector<cytnx_uint64>{3}));
        EXPECT_DOUBLE_EQ(vec.at<double>({0}), 2.0);
        EXPECT_DOUBLE_EQ(vec.at<double>({1}), 3.0);
        EXPECT_DOUBLE_EQ(vec.at<double>({2}), 4.0);

        EXPECT_THROW(scalar += vec, std::logic_error);

        Tensor legacy_shape_one({1}, Type.Double);
        legacy_shape_one.at<double>({0}) = 2.0;
        EXPECT_FALSE(legacy_shape_one.is_scalar());
        EXPECT_DOUBLE_EQ(legacy_shape_one.item<double>(), 2.0);
        Tensor legacy_shape_one_one({1, 1}, Type.Double);
        legacy_shape_one_one.at<double>({0, 0}) = 3.0;
        EXPECT_FALSE(legacy_shape_one_one.is_scalar());
        EXPECT_DOUBLE_EQ(legacy_shape_one_one.item<double>(), 3.0);

        out = legacy_shape_one + vec;
        EXPECT_EQ(out.shape(), vec.shape());
        EXPECT_DOUBLE_EQ(out.at<double>({0}), 4.0);
        EXPECT_DOUBLE_EQ(out.at<double>({1}), 5.0);
        EXPECT_DOUBLE_EQ(out.at<double>({2}), 6.0);

        out = vec + legacy_shape_one_one;
        EXPECT_EQ(out.shape(), vec.shape());
        EXPECT_DOUBLE_EQ(out.at<double>({0}), 5.0);
        EXPECT_DOUBLE_EQ(out.at<double>({1}), 6.0);
        EXPECT_DOUBLE_EQ(out.at<double>({2}), 7.0);

        out = legacy_shape_one + legacy_shape_one_one;
        EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{1, 1}));
        EXPECT_DOUBLE_EQ(out.at<double>({0, 0}), 5.0);

        vec += legacy_shape_one;
        EXPECT_DOUBLE_EQ(vec.at<double>({0}), 4.0);
        EXPECT_DOUBLE_EQ(vec.at<double>({1}), 5.0);
        EXPECT_DOUBLE_EQ(vec.at<double>({2}), 6.0);

        vec -= legacy_shape_one_one;
        vec *= legacy_shape_one;
        vec /= legacy_shape_one;
        EXPECT_DOUBLE_EQ(vec.at<double>({0}), 1.0);
        EXPECT_DOUBLE_EQ(vec.at<double>({1}), 2.0);
        EXPECT_DOUBLE_EQ(vec.at<double>({2}), 3.0);
      }

      TEST_F(TensorTest, RankZeroVectordotIsScalar) {
        Tensor vec = arange(0, 3, 1, Type.Double);

        Tensor dot = linalg::Vectordot(vec, vec, false);
        EXPECT_TRUE(dot.is_scalar());
        EXPECT_EQ(dot.shape().size(), 0);
        EXPECT_DOUBLE_EQ(dot.item<double>(), 5.0);

        Tensor scaled = dot * vec;
        EXPECT_EQ(scaled.shape(), (std::vector<cytnx_uint64>{3}));
        EXPECT_DOUBLE_EQ(scaled.at<double>({0}), 0.0);
        EXPECT_DOUBLE_EQ(scaled.at<double>({1}), 5.0);
        EXPECT_DOUBLE_EQ(scaled.at<double>({2}), 10.0);
      }

      TEST_F(TensorTest, RankZeroReductionsAreScalarAndBroadcast) {
        Tensor vec = arange(1, 4, 1, Type.Double);

        Tensor sum = linalg::Sum(vec);
        EXPECT_TRUE(sum.is_scalar());
        EXPECT_DOUBLE_EQ(sum.item<double>(), 6.0);

        Tensor max = linalg::Max(vec);
        EXPECT_TRUE(max.is_scalar());
        EXPECT_DOUBLE_EQ(max.item<double>(), 3.0);

        Tensor min = linalg::Min(vec);
        EXPECT_TRUE(min.is_scalar());
        EXPECT_DOUBLE_EQ(min.item<double>(), 1.0);

        Tensor shifted = sum + vec;
        EXPECT_EQ(shifted.shape(), (std::vector<cytnx_uint64>{3}));
        EXPECT_DOUBLE_EQ(shifted.at<double>({0}), 7.0);
        EXPECT_DOUBLE_EQ(shifted.at<double>({1}), 8.0);
        EXPECT_DOUBLE_EQ(shifted.at<double>({2}), 9.0);
      }

      TEST_F(TensorTest, RankZeroBroadcastComparison) {
        Tensor scalar({}, Type.Double);
        scalar.item<double>() = 2.0;
        Tensor vec = arange(0, 3, 1, Type.Double);

        Tensor out = linalg::Cpr(scalar, vec);
        EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
        EXPECT_FALSE(out.at<cytnx_bool>({0}));
        EXPECT_FALSE(out.at<cytnx_bool>({1}));
        EXPECT_TRUE(out.at<cytnx_bool>({2}));
      }

      TEST_F(TensorTest, VoidTensorArithmeticThrowsControlledError) {
        Tensor uninitialized;
        Tensor scalar({}, Type.Double);
        scalar.item<double>() = 2.0;
        Tensor vec = arange(0, 3, 1, Type.Double);

        EXPECT_THROW((void)(uninitialized + scalar), std::logic_error);
        EXPECT_THROW((void)(scalar + uninitialized), std::logic_error);
        EXPECT_THROW((void)(uninitialized + vec), std::logic_error);

        const auto expect_scalar_error = [](auto operation, const std::string &op_name) {
          try {
            operation();
            FAIL() << "expected " << op_name << " with an uninitialized Tensor to throw";
          } catch (const std::logic_error &error) {
            const std::string message = error.what();
            EXPECT_NE(message.find(op_name), std::string::npos);
            EXPECT_NE(message.find("uninitialized Tensor"), std::string::npos);
          }
        };

        expect_scalar_error([&] { (void)(2.0 + uninitialized); }, "Add");
        expect_scalar_error([&] { (void)(uninitialized + 2.0); }, "Add");
        expect_scalar_error([&] { (void)(2.0 - uninitialized); }, "Sub");
        expect_scalar_error([&] { (void)(uninitialized - 2.0); }, "Sub");
        expect_scalar_error([&] { (void)(2.0 * uninitialized); }, "Mul");
        expect_scalar_error([&] { (void)(uninitialized * 2.0); }, "Mul");
        expect_scalar_error([&] { (void)(2.0 / uninitialized); }, "Div");
        expect_scalar_error([&] { (void)(uninitialized / 2.0); }, "Div");
        expect_scalar_error([&] { (void)(2.0 % uninitialized); }, "Mod");
        expect_scalar_error([&] { (void)(uninitialized % 2.0); }, "Mod");

        const Scalar scalar_value(2.0);
        expect_scalar_error([&] { (void)(scalar_value - uninitialized); }, "Sub");
        expect_scalar_error([&] { (void)(uninitialized - scalar_value); }, "Sub");
      }

      TEST_F(TensorTest, RankZeroAppendAsScalarElement) {
        Tensor vec = arange(0, 2, 1, Type.Double);
        Tensor scalar({}, Type.Double);
        scalar.item<double>() = 3.5;

        vec.append(scalar);
        EXPECT_EQ(vec.shape(), (std::vector<cytnx_uint64>{3}));
        EXPECT_DOUBLE_EQ(vec.at<double>({0}), 0.0);
        EXPECT_DOUBLE_EQ(vec.at<double>({1}), 1.0);
        EXPECT_DOUBLE_EQ(vec.at<double>({2}), 3.5);

        Tensor uninitialized;
        EXPECT_THROW(vec.append(uninitialized), std::logic_error);
        EXPECT_THROW(scalar.append(scalar), std::logic_error);

        Tensor matrix = zeros({1, 2}, Type.Double);
        Storage row(2, Type.Double);
        row.at<double>(0) = 4.0;
        row.at<double>(1) = 5.0;
        matrix.append(row);
        EXPECT_EQ(matrix.shape(), (std::vector<cytnx_uint64>{2, 2}));
        EXPECT_DOUBLE_EQ(matrix.at<double>({1, 0}), 4.0);
        EXPECT_DOUBLE_EQ(matrix.at<double>({1, 1}), 5.0);

        Storage empty;
        EXPECT_THROW(matrix.append(empty), std::logic_error);
        EXPECT_THROW(scalar.append(row), std::logic_error);
        EXPECT_THROW(uninitialized.append(row), std::logic_error);
      }

      TEST_F(TensorTest, RankZeroSortReturnsScalar) {
        Tensor scalar({}, Type.Double);
        scalar.item<double>() = 7.5;

        Tensor sorted = algo::Sort(scalar);
        EXPECT_TRUE(sorted.is_scalar());
        EXPECT_DOUBLE_EQ(sorted.item<double>(), 7.5);
      }

      TEST_F(TensorTest, RankZeroDiagThrowsControlledError) {
        Tensor scalar({}, Type.Double);
        scalar.item<double>() = 2.0;
        EXPECT_THROW(linalg::Diag(scalar), std::logic_error);
      }

      TEST_F(TensorTest, RankZeroSaveLoad) {
        Tensor scalar({}, Type.Double);
        scalar.item<double>() = 8.25;

        const auto path = std::filesystem::temp_directory_path() / "cytnx_rank_zero_tensor.cytn";
        scalar.Save(path.string());

        {
          std::ifstream header(path, std::ios::binary);
          ASSERT_TRUE(header.is_open());
          unsigned int magic = 0;
          unsigned int version = 0;
          header.read(reinterpret_cast<char *>(&magic), sizeof(magic));
          header.read(reinterpret_cast<char *>(&version), sizeof(version));
          EXPECT_EQ(magic, 889);
          EXPECT_EQ(version, 1);
        }

        Tensor loaded = Tensor::Load(path.string());
        EXPECT_EQ(loaded.shape().size(), 0);
        EXPECT_EQ(loaded.rank(), 0);
        EXPECT_EQ(loaded.dtype(), Type.Double);
        EXPECT_DOUBLE_EQ(loaded.item<double>(), 8.25);
        std::filesystem::remove(path);

        Tensor vector_one({1}, Type.Double);
        vector_one.at<double>({0}) = 3.5;
        const auto vector_path =
          std::filesystem::temp_directory_path() / "cytnx_rank_one_size_one_tensor.cytn";
        vector_one.Save(vector_path.string());

        Tensor loaded_vector_one = Tensor::Load(vector_path.string());
        EXPECT_EQ(loaded_vector_one.shape(), (std::vector<cytnx_uint64>{1}));
        EXPECT_FALSE(loaded_vector_one.is_scalar());
        EXPECT_DOUBLE_EQ(loaded_vector_one.at<double>({0}), 3.5);
        std::filesystem::remove(vector_path);
      }

      TEST_F(TensorTest, Identity) {
        Tensor tn = identity(2, Type.Double, Device.cpu);
        EXPECT_EQ(tn.shape().size(), 2);
        EXPECT_EQ(tn.shape()[0], 2);
        EXPECT_EQ(tn.shape()[1], 2);
        EXPECT_EQ(tn.is_contiguous(), true);
        EXPECT_EQ(tn.dtype(), Type.Double);
        EXPECT_EQ(tn.device(), Device.cpu);
        EXPECT_DOUBLE_EQ((double)tn(0, 0).item().real(), 1);
        EXPECT_DOUBLE_EQ((double)tn(1, 1).item().real(), 1);
        EXPECT_DOUBLE_EQ((double)tn(0, 1).item().real(), 0);
        EXPECT_DOUBLE_EQ((double)tn(1, 0).item().real(), 0);

        tn = identity(3, Type.Double, Device.cpu);
        EXPECT_EQ(tn.shape().size(), 2);
        EXPECT_EQ(tn.shape()[0], 3);
        EXPECT_EQ(tn.shape()[1], 3);
        EXPECT_EQ(tn.is_contiguous(), true);
        EXPECT_EQ(tn.dtype(), Type.Double);
        EXPECT_EQ(tn.device(), Device.cpu);
        EXPECT_DOUBLE_EQ(tn.at<double>({0, 0}), 1);
        EXPECT_DOUBLE_EQ(tn.at<double>({1, 1}), 1);
        EXPECT_DOUBLE_EQ(tn.at<double>({0, 1}), 0);
        EXPECT_DOUBLE_EQ(tn.at<double>({1, 0}), 0);
        EXPECT_DOUBLE_EQ(tn.at<double>({2, 0}), 0);
        EXPECT_DOUBLE_EQ(tn.at<double>({2, 1}), 0);
        EXPECT_DOUBLE_EQ(tn.at<double>({0, 2}), 0);
        EXPECT_DOUBLE_EQ(tn.at<double>({1, 2}), 0);
        EXPECT_DOUBLE_EQ(tn.at<double>({2, 2}), 1);
      }

      TEST_F(TensorTest, Eye) {
        Tensor tn = eye(2, Type.Double, Device.cpu);
        EXPECT_EQ(tn.shape().size(), 2);
        EXPECT_EQ(tn.shape()[0], 2);
        EXPECT_EQ(tn.shape()[1], 2);
        EXPECT_EQ(tn.is_contiguous(), true);
        EXPECT_EQ(tn.dtype(), Type.Double);
        EXPECT_EQ(tn.device(), Device.cpu);
        EXPECT_DOUBLE_EQ((double)tn(0, 0).item().real(), 1);
        EXPECT_DOUBLE_EQ((double)tn(1, 1).item().real(), 1);
        EXPECT_DOUBLE_EQ((double)tn(0, 1).item().real(), 0);
        EXPECT_DOUBLE_EQ((double)tn(1, 0).item().real(), 0);

        tn = eye(3, Type.Double, Device.cpu);
        EXPECT_EQ(tn.shape().size(), 2);
        EXPECT_EQ(tn.shape()[0], 3);
        EXPECT_EQ(tn.shape()[1], 3);
        EXPECT_EQ(tn.is_contiguous(), true);
        EXPECT_EQ(tn.dtype(), Type.Double);
        EXPECT_EQ(tn.device(), Device.cpu);
        EXPECT_DOUBLE_EQ(tn.at<double>({0, 0}), 1);
        EXPECT_DOUBLE_EQ(tn.at<double>({1, 1}), 1);
        EXPECT_DOUBLE_EQ(tn.at<double>({0, 1}), 0);
        EXPECT_DOUBLE_EQ(tn.at<double>({1, 0}), 0);
        EXPECT_DOUBLE_EQ(tn.at<double>({2, 0}), 0);
        EXPECT_DOUBLE_EQ(tn.at<double>({2, 1}), 0);
        EXPECT_DOUBLE_EQ(tn.at<double>({0, 2}), 0);
        EXPECT_DOUBLE_EQ(tn.at<double>({1, 2}), 0);
        EXPECT_DOUBLE_EQ(tn.at<double>({2, 2}), 1);
      }

      // norm() (#676): returns a Scalar carrying the tensor's own precision, equal in value to
      // the deprecated Norm().item(), on a known 3-4-5 vector.
      TEST_F(TensorTest, Norm) {
        Tensor v({2}, Type.Double);
        v(0) = 3.0;
        v(1) = 4.0;
        double n = double(v.norm());
        EXPECT_DOUBLE_EQ(n, 5.0);
        EXPECT_DOUBLE_EQ(n, double(v.Norm().item().real()));

        // norm() returns a Scalar carrying the tensor's precision (#1000 review, ianmccul):
        // Float stays Float, so x /= x.norm() no longer silently promotes Float to Double.
        EXPECT_EQ(v.norm().dtype(), Type.Double);
        Tensor vf = v.astype(Type.Float);
        EXPECT_EQ(vf.norm().dtype(), Type.Float);
        vf /= vf.norm();
        EXPECT_EQ(vf.dtype(), Type.Float);
      }
      // TEST_F(TensorTest, ApproxEq) {
      //   User_debug = true;
      //   EXPECT_TRUE(tar3456.approx_eq(tar3456));
      //   EXPECT_FALSE(tar345.approx_eq(tar345.permute(1, 0, 2)));
      //   EXPECT_FALSE(tar3456.approx_eq(tarcomplex3456));
      //   EXPECT_TRUE(tone3456.approx_eq(tone3456.astype(Type.ComplexFloat), 1e-5));
      // }

      TEST(Tensor, ItemDtypeMismatchThrows) {
        Tensor t = zeros({1}, Type.Float);
        EXPECT_THROW(t.item<double>(), std::logic_error);
        EXPECT_NO_THROW(t.item<float>());
      }

      TEST(Tensor, ReshapeRejectsMultipleUnknownDims) {
        Tensor t = zeros({12}, Type.Double);
        EXPECT_THROW(t.reshape({-1, -1}), std::logic_error);
        EXPECT_THROW(t.reshape_({-1, -1}), std::logic_error);
        EXPECT_THROW(t.reshape({-2, 6}), std::logic_error);
        // a single -1 must keep working
        Tensor r = t.reshape({3, -1});
        EXPECT_EQ(r.shape(), (std::vector<cytnx_uint64>{3, 4}));
      }

      TEST(Tensor, ReshapeRejectsZeroDimWithUnknownDim) {
        Tensor t = zeros({12}, Type.Double);
        // new_N == 0 previously fed a modulo/division by zero (UB / SIGFPE on x86)
        EXPECT_THROW(t.reshape({0, -1}), std::logic_error);
        EXPECT_THROW(t.reshape_({0, -1}), std::logic_error);
        EXPECT_THROW(t.reshape_({-1, 0}), std::logic_error);
      }

      TEST(Tensor, FailedReshapeLeavesShapeUnchanged) {
        Tensor t = zeros({12}, Type.Double);
        EXPECT_THROW(t.reshape_({7, 2}), std::logic_error);
        EXPECT_EQ(t.shape(), (std::vector<cytnx_uint64>{12}));
        EXPECT_THROW(t.reshape_({5, -1}), std::logic_error);
        EXPECT_EQ(t.shape(), (std::vector<cytnx_uint64>{12}));
        EXPECT_THROW(t.reshape_({0, -1}), std::logic_error);
        EXPECT_EQ(t.shape(), (std::vector<cytnx_uint64>{12}));
      }

      // NOTE: `Tensor b = a;` makes `b` an alias of the *same* Tensor_impl as `a`
      // (Tensor's copy ctor just copies the intrusive_ptr), so `is(a.storage(),
      // b.storage())` would trivially read the same Storage field twice and could
      // never observe the #906 detach bug. To exercise a genuine "two independent
      // Tensor handles sharing one Storage" scenario (e.g. what a view/slice would
      // produce), we build `b` via Tensor::from_storage(a.storage()), which creates
      // a brand-new Tensor_impl whose _storage is a copy of the Storage handle
      // (same underlying Storage_base, distinct Tensor_impl).
      TEST(Tensor, ScalarInplaceAddKeepsStorageSharing) {
        Tensor a = zeros({4}, Type.Double);
        Tensor b = Tensor::from_storage(a.storage());  // distinct Tensor_impl, shared Storage
        ASSERT_TRUE(is(a.storage(), b.storage()));
        a += 1.0;
        EXPECT_TRUE(is(a.storage(), b.storage()));
        EXPECT_DOUBLE_EQ(b.storage().at<double>(0), 1.0);
      }

      TEST(Tensor, ScalarInplaceOpsPreserveDtype) {
        Tensor a = ones({2}, Type.Float);
        a += 1.0;  // double scalar must not promote the tensor
        a -= 0.5;
        a *= 2.0;
        a /= 3.0;
        EXPECT_EQ(a.dtype(), Type.Float);
        EXPECT_FLOAT_EQ(a.storage().at<float>(0), 1.0f);
      }

      TEST(Tensor, ScalarInplaceRealPlusComplexThrows) {
        Tensor a = zeros({2}, Type.Double);
        EXPECT_THROW(a += cytnx_complex128(0, 1), std::logic_error);
        EXPECT_THROW(a -= cytnx_complex128(0, 1), std::logic_error);
        EXPECT_THROW(a *= cytnx_complex128(0, 1), std::logic_error);
        EXPECT_THROW(a /= cytnx_complex128(0, 1), std::logic_error);
      }

      TEST(Tensor, ScalarInplaceIntTensorTruncatesFractionalScalar) {
        Tensor a = ones({2}, Type.Int64);
        a += 2.7;  // was: promoted to Double (3.7); now: stays Int64, truncates
        EXPECT_EQ(a.dtype(), Type.Int64);
        EXPECT_EQ(a.storage().at<cytnx_int64>(0), 3);
      }

      // Mirrors the actual #906 report through public API: permute() produces a
      // distinct Tensor_impl sharing the same Storage (Tensor_impl::permute does
      // `out->_storage = this->_storage`) flagged non-contiguous, so this also
      // exercises the non-contiguous scalar broadcast path of iAdd.
      TEST(Tensor, ScalarInplaceOnPermutedViewMutatesSharedStorage) {
        Tensor a = zeros({2, 3}, Type.Double);
        Tensor v = a.permute({1, 0});  // distinct impl, shared storage, non-contiguous
        ASSERT_FALSE(v.is_contiguous());
        ASSERT_TRUE(is(a.storage(), v.storage()));
        v += 1.0;
        EXPECT_TRUE(is(a.storage(), v.storage()));
        EXPECT_DOUBLE_EQ(a.storage().at<double>(0), 1.0);
      }

      TEST(Tensor, ScalarInplaceSubMulDivKeepStorageSharing) {
        Tensor a = ones({3}, Type.Double);
        Tensor b = Tensor::from_storage(a.storage());
        a -= 0.5;
        a *= 4.0;
        a /= 2.0;
        EXPECT_TRUE(is(a.storage(), b.storage()));
        EXPECT_DOUBLE_EQ(b.storage().at<double>(2), 1.0);
      }

      TEST(Tensor, CytnxScalarInplaceAddKeepsStorageSharing) {
        Tensor a = zeros({2}, Type.Double);
        Tensor b = Tensor::from_storage(a.storage());
        a += Scalar(2.5);
        EXPECT_TRUE(is(a.storage(), b.storage()));
        EXPECT_DOUBLE_EQ(b.storage().at<double>(1), 2.5);
      }

      TEST(Tensor, ScalarInplaceRealTimesComplexErrorNamesOperator) {
        Tensor a = zeros({2}, Type.Double);
        try {
          a *= cytnx_complex128(0, 1);
          FAIL() << "expected real *= complex to throw";
        } catch (const std::logic_error &e) {
          EXPECT_NE(std::string(e.what()).find("*="), std::string::npos) << e.what();
        }
        try {
          a /= cytnx_complex128(0, 1);
          FAIL() << "expected real /= complex to throw";
        } catch (const std::logic_error &e) {
          EXPECT_NE(std::string(e.what()).find("/="), std::string::npos) << e.what();
        }
      }

      // --- #941 mixed-dtype arithmetic regression tests ---
      //
      // Motivation (#941): a dispatch table can select a kernel whose C++ output
      // type is e.g. double, while the actual output storage object is still
      // StorageImplementation<int16_t> -- writing promoted/floating results into
      // narrower/integer storage. These tests pin the CORRECT behavior: in-place
      // arithmetic must promote dtype exactly like the equivalent out-of-place
      // operation (never truncate into the original lhs storage type), and Div
      // must implement Python true-division semantics.

      TEST(MixedDtypeArithmetic, InplaceSubIntMinusDoublePromotes) {
        Tensor Lt = Tensor({3}, Type.Int16, Device.cpu);
        Lt.at<cytnx_int16>({0}) = 10;
        Lt.at<cytnx_int16>({1}) = 20;
        Lt.at<cytnx_int16>({2}) = 30;
        Tensor Rt = Tensor({3}, Type.Double, Device.cpu);
        Rt.at<cytnx_double>({0}) = 0.5;
        Rt.at<cytnx_double>({1}) = 0.5;
        Rt.at<cytnx_double>({2}) = 0.5;

        linalg::iSub(Lt, Rt);

        EXPECT_EQ(Lt.dtype(), (unsigned int)Type.Double)
          << "Int16 -= Double must promote Lt's storage to Double, not truncate into Int16";
        EXPECT_DOUBLE_EQ(Lt.at<cytnx_double>({0}), 9.5);
        EXPECT_DOUBLE_EQ(Lt.at<cytnx_double>({1}), 19.5);
        EXPECT_DOUBLE_EQ(Lt.at<cytnx_double>({2}), 29.5);
      }

      TEST(MixedDtypeArithmetic, InplaceAddIntPlusDoublePromotes) {
        Tensor Lt = Tensor({2}, Type.Int32, Device.cpu);
        Lt.at<cytnx_int32>({0}) = 1;
        Lt.at<cytnx_int32>({1}) = 2;
        Tensor Rt = Tensor({2}, Type.Double, Device.cpu);
        Rt.at<cytnx_double>({0}) = 1.25;
        Rt.at<cytnx_double>({1}) = 2.25;

        linalg::iAdd(Lt, Rt);

        EXPECT_EQ(Lt.dtype(), (unsigned int)Type.Double);
        EXPECT_DOUBLE_EQ(Lt.at<cytnx_double>({0}), 2.25);
        EXPECT_DOUBLE_EQ(Lt.at<cytnx_double>({1}), 4.25);
      }

      TEST(MixedDtypeArithmetic, InplaceMulIntTimesDoublePromotes) {
        Tensor Lt = Tensor({2}, Type.Int64, Device.cpu);
        Lt.at<cytnx_int64>({0}) = 3;
        Lt.at<cytnx_int64>({1}) = 4;
        Tensor Rt = Tensor({2}, Type.Double, Device.cpu);
        Rt.at<cytnx_double>({0}) = 1.5;
        Rt.at<cytnx_double>({1}) = 2.5;

        linalg::iMul(Lt, Rt);

        EXPECT_EQ(Lt.dtype(), (unsigned int)Type.Double);
        EXPECT_DOUBLE_EQ(Lt.at<cytnx_double>({0}), 4.5);
        EXPECT_DOUBLE_EQ(Lt.at<cytnx_double>({1}), 10.0);
      }

      TEST(MixedDtypeArithmetic, DivTrueDivisionIntOverIntProducesDouble) {
        Tensor Lt = Tensor({1}, Type.Int64, Device.cpu);
        Lt.at<cytnx_int64>({0}) = 3;
        Tensor Rt = Tensor({1}, Type.Int64, Device.cpu);
        Rt.at<cytnx_int64>({0}) = 2;

        Tensor out = linalg::Div(Lt, Rt);

        EXPECT_EQ(out.dtype(), (unsigned int)Type.Double)
          << "Int64/Int64 must produce a floating-point result (Python true-division semantics)";
        EXPECT_DOUBLE_EQ(out.at<cytnx_double>({0}), 1.5);
      }

      TEST(MixedDtypeArithmetic, IDivTrueDivisionPromotesLhsToDouble) {
        Tensor Lt = Tensor({1}, Type.Int16, Device.cpu);
        Lt.at<cytnx_int16>({0}) = 3;
        Tensor Rt = Tensor({1}, Type.Double, Device.cpu);
        Rt.at<cytnx_double>({0}) = 2.0;

        linalg::iDiv(Lt, Rt);

        EXPECT_EQ(Lt.dtype(), (unsigned int)Type.Double);
        EXPECT_DOUBLE_EQ(Lt.at<cytnx_double>({0}), 1.5);
      }

      TEST(MixedDtypeArithmetic, IDivInt64OverDoublePromotes) {
        Tensor Lt = Tensor({1}, Type.Int64, Device.cpu);
        Lt.at<cytnx_int64>({0}) = 3;
        Tensor Rt = Tensor({1}, Type.Double, Device.cpu);
        Rt.at<cytnx_double>({0}) = 2.0;
        linalg::iDiv(Lt, Rt);
        EXPECT_EQ(Lt.dtype(), (unsigned int)Type.Double);
        EXPECT_DOUBLE_EQ(Lt.at<cytnx_double>({0}), 1.5);
      }

      TEST(MixedDtypeArithmetic, IDivInt32OverDoublePromotes) {
        Tensor Lt = Tensor({1}, Type.Int32, Device.cpu);
        Lt.at<cytnx_int32>({0}) = 3;
        Tensor Rt = Tensor({1}, Type.Double, Device.cpu);
        Rt.at<cytnx_double>({0}) = 2.0;
        linalg::iDiv(Lt, Rt);
        EXPECT_EQ(Lt.dtype(), (unsigned int)Type.Double);
        EXPECT_DOUBLE_EQ(Lt.at<cytnx_double>({0}), 1.5);
      }

      TEST(MixedDtypeArithmetic, IDivUint16OverDoublePromotes) {
        Tensor Lt = Tensor({1}, Type.Uint16, Device.cpu);
        Lt.at<cytnx_uint16>({0}) = 3;
        Tensor Rt = Tensor({1}, Type.Double, Device.cpu);
        Rt.at<cytnx_double>({0}) = 2.0;
        linalg::iDiv(Lt, Rt);
        EXPECT_EQ(Lt.dtype(), (unsigned int)Type.Double);
        EXPECT_DOUBLE_EQ(Lt.at<cytnx_double>({0}), 1.5);
      }

      TEST(MixedDtypeArithmetic, OutOfPlaceAddMixedComplexReal) {
        // Sanity check: the already-working out-of-place complex/real promotion
        // path is unaffected by the typed-storage conversion.
        Tensor Lt = Tensor({1}, Type.ComplexFloat, Device.cpu);
        Lt.at<cytnx_complex64>({0}) = cytnx_complex64(1.0f, 1.0f);
        Tensor Rt = Tensor({1}, Type.Double, Device.cpu);
        Rt.at<cytnx_double>({0}) = 2.0;
        Tensor out = linalg::Add(Lt, Rt);
        EXPECT_EQ(out.dtype(), (unsigned int)Type.ComplexDouble);
      }

      TEST(MixedDtypeArithmetic, CprMixedIntDoubleComparesCorrectly) {
        Tensor Lt = Tensor({3}, Type.Int32, Device.cpu);
        Lt.at<cytnx_int32>({0}) = 1;
        Lt.at<cytnx_int32>({1}) = 2;
        Lt.at<cytnx_int32>({2}) = 3;
        Tensor Rt = Tensor({3}, Type.Double, Device.cpu);
        Rt.at<cytnx_double>({0}) = 1.0;
        Rt.at<cytnx_double>({1}) = 2.5;
        Rt.at<cytnx_double>({2}) = 3.0;

        Tensor out = linalg::Cpr(Lt, Rt);

        EXPECT_EQ(out.dtype(), (unsigned int)Type.Bool);
        EXPECT_EQ(out.at<cytnx_bool>({0}), true);
        EXPECT_EQ(out.at<cytnx_bool>({1}), false);
        EXPECT_EQ(out.at<cytnx_bool>({2}), true);
      }

      TEST(MixedDtypeArithmetic, ModMixedIntDoublePromotes) {
        Tensor Lt = Tensor({1}, Type.Int32, Device.cpu);
        Lt.at<cytnx_int32>({0}) = 7;
        Tensor Rt = Tensor({1}, Type.Double, Device.cpu);
        Rt.at<cytnx_double>({0}) = 2.5;

        Tensor out = linalg::Mod(Lt, Rt);

        EXPECT_EQ(out.dtype(), (unsigned int)Type.Double);
        EXPECT_DOUBLE_EQ(out.at<cytnx_double>({0}), std::fmod(7.0, 2.5));
      }

    }  // namespace
  }  // namespace test
}  // namespace cytnx
