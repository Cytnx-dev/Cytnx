#include "stat.hpp"
#include "Type.hpp"

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  namespace stat {

    Histogram::Histogram(const unsigned long long &Nbins, const double &min_val,
                         const double &max_val) {
      cytnx_error_msg(min_val >= max_val, "[ERROR] Cannot have min >= max%s", "\n");
      this->min = min_val;
      this->max = max_val;
      this->bins = Nbins;

      this->vars = Storage(bins, Type.Double);
      this->x = Storage(bins, Type.Double);
      double dx = double(max_val - min_val) / Nbins;
      for (unsigned int i = 0; i < x.size(); i++) {
        this->x.at<double>(i) = dx * i;
      }

      total_count = 0;
    }

    void Histogram::normalize() {
      // get the density.
      double dx = double(max - min) / bins;
      double w = 1. / total_count / dx;

      // wrapping around storage and use Tensor API
      Tensor tmp = Tensor::from_storage(this->vars);
      tmp *= w;
      this->vars = tmp.storage();
    }

    void Histogram::print(std::ostream &os) const {
      os << "[Histogram 1D] Real" << std::endl;
      os << "Nbins: " << this->bins << std::endl;
      os << "bound: [ " << this->min << " , " << this->max << " ]\n";
      os << "current count: " << this->total_count << std::endl;
    }

    //-----------[2d]
    Histogram2d::Histogram2d(const unsigned long long &Nbinx, const unsigned long long &Nbiny,
                             const double &min_x, const double &max_x, const double &min_y,
                             const double &max_y) {
      cytnx_error_msg(min_x >= max_x, "[ERROR] Cannot have min >= max [x axis]%s", "\n");
      cytnx_error_msg(min_y >= max_y, "[ERROR] Cannot have min >= max [y axis]%s", "\n");
      this->minx = min_x;
      this->maxx = max_x;
      this->miny = min_y;
      this->maxy = max_y;

      this->binx = Nbinx;
      this->biny = Nbiny;

      this->vars = Storage(Nbinx * Nbiny, Type.Double);
      this->x = Storage(Nbinx, Type.Double);
      this->y = Storage(Nbiny, Type.Double);

      double dx = double(max_x - min_x) / Nbinx;
      for (unsigned int i = 0; i < x.size(); i++) {
        this->x.at<double>(i) = dx * i;
      }
      double dy = double(max_y - min_y) / Nbiny;
      for (unsigned int i = 0; i < y.size(); i++) {
        this->y.at<double>(i) = dy * i;
      }
      total_count = 0;
    }

    void Histogram2d::normalize() {
      // get the density.
      double dx = double(maxx - minx) / binx;
      double dy = double(maxy - miny) / biny;
      double w = 1. / (total_count * dx * dy);

      // wrapping around storage and use Tensor API
      Tensor tmp = Tensor::from_storage(this->vars);
      tmp *= w;
      this->vars = tmp.storage();
    }

    void Histogram2d::print(std::ostream &os) const {
      os << "[Histogram 2D] Real" << std::endl;
      os << "Nbins: [x= " << this->binx << " , y= " << this->biny << " ]\n";
      os << "bound,x: [ " << this->minx << " , " << this->maxx << " ]\n";
      os << "bound,y: [ " << this->miny << " , " << this->maxy << " ]\n";
      os << "current count: " << this->total_count << std::endl;
    }

  }  // namespace stat
}  // namespace cytnx
#endif
