std::vector<Scalar> out;

out.push_back(Scalar(1.33));  // double
out.push_back(Scalar(10));  // int
out.push_back(Scalar(cytnx_complex128(3, 4)));  // complex double

std::cout << out[0] << out[1] << out[2] << std::endl;
