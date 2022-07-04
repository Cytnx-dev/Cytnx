#ifndef _stat_H_
#define _stat_H_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Storage.hpp"
#include "Tensor.hpp"
#include <algorithm>
#include <iostream>

namespace cytnx {
  namespace stat {

    /// 1D, real value histogram
    class Histogram {
     public:
      double min;
      double max;
      uint64_t bins;
      cytnx::Storage vars;
      cytnx::Storage x;

      // std::vector<double> vars;
      // std::vector<double> x;

      double total_count;

      /**
      @brief initialize a histogram
      */
      Histogram(const unsigned long long &Nbins, const double &min_val, const double &max_val);

      ///@cond
      Histogram(const Histogram &rhs) {
        this->min = rhs.min;
        this->max = rhs.max;
        this->bins = rhs.bins;
        this->vars = rhs.vars.clone();
        this->x = rhs.x.clone();
        this->total_count = rhs.total_count;
      }

      Histogram &operator=(const Histogram &rhs) {
        this->min = rhs.min;
        this->max = rhs.max;
        this->bins = rhs.bins;
        this->vars = rhs.vars.clone();
        this->x = rhs.x.clone();
        this->total_count = rhs.total_count;
        return *this;
      }
      ///@endcond

      void clear_vars() {
        total_count = 0;
        memset(this->vars.data(), 0, sizeof(double) * this->vars.size());
      }

      template <class T>
      void accumulate(const std::vector<T> &data) {
        std::vector<T> tmp = data;
        std::sort(tmp.begin(), tmp.end());

        uint64_t cntr = 0;
        double curr_x = 0;
        double dx = double(max - min) / bins;

        // for each elem in data, compare
        for (unsigned long long i = 0; i < tmp.size(); i++) {
          while (cntr <= this->bins) {
            if (tmp[i] < curr_x) {
              if (cntr) {
                vars.at<double>(cntr - 1) += 1;
                total_count += 1;
              }
              break;
            }
            cntr++;
            curr_x = cntr * dx;
          }
        }
      }

      void normalize();
      void print() const;

      const Storage &get_x() const {
        // get x
        return this->x;
      }
      Storage &get_x() {
        // get x
        return this->x;
      }

      cytnx_uint64 size() { return this->x.size(); }

    };  // class histogram

    /// 2D, real value histogram
    class Histogram2d {
     public:
      double minx, miny;
      double maxx, maxy;
      uint64_t binx, biny;
      cytnx::Storage vars;
      cytnx::Storage x;
      cytnx::Storage y;

      // std::vector<double> vars;
      // std::vector<double> x;

      double total_count;

      /**
      @brief initialize a histogram
      */
      Histogram2d(const unsigned long long &Nbinx, const unsigned long long &Nbiny,
                  const double &min_x, const double &max_x, const double &min_y,
                  const double &max_y);

      ///@cond
      Histogram2d(const Histogram2d &rhs) {
        this->minx = rhs.minx;
        this->maxx = rhs.maxx;
        this->miny = rhs.miny;
        this->maxy = rhs.maxy;
        this->binx = rhs.binx;
        this->biny = rhs.biny;
        this->vars = rhs.vars.clone();
        this->x = rhs.x.clone();
        this->y = rhs.y.clone();
        this->total_count = rhs.total_count;
      }

      Histogram2d &operator=(const Histogram2d &rhs) {
        this->minx = rhs.minx;
        this->maxx = rhs.maxx;
        this->miny = rhs.miny;
        this->maxy = rhs.maxy;
        this->binx = rhs.binx;
        this->biny = rhs.biny;
        this->vars = rhs.vars.clone();
        this->x = rhs.x.clone();
        this->y = rhs.y.clone();
        this->total_count = rhs.total_count;
        return *this;
      }
      ///@endcond

      void clear_vars() {
        total_count = 0;
        memset(this->vars.data(), 0, sizeof(double) * this->vars.size());
      }

      template <class T>
      void accumulate(const std::vector<T> &data_x, const std::vector<T> &data_y) {
        //[not fin!]
        cytnx_error_msg(
          data_x.size() != data_y.size(),
          "[ERROR][Histogram2d::accumulate] data_x and data_y should have same size!%s", "\n");

        double dx = double(maxx - minx) / binx;
        double dy = double(maxy - miny) / biny;

        unsigned int nx, ny;

        // for each elem in data, compare
        for (unsigned long long i = 0; i < data_x.size(); i++) {
          nx = floor(data_x[i] / dx);
          ny = floor(data_y[i] / dy);
          vars.at<double>(ny * binx + nx) += 1;
          total_count += 1;
        }
      }

      void normalize();
      void print() const;

      const Storage &get_x() const {
        // get x
        return this->x;
      }
      Storage &get_x() {
        // get x
        return this->x;
      }

      const Storage &get_y() const {
        // get y
        return this->y;
      }
      Storage &get_y() {
        // get y
        return this->y;
      }

      std::vector<cytnx_uint64> size() {
        return std::vector<cytnx_uint64>({this->x.size(), this->y.size()});
      }

    };  // class histogram

    // function, statistical:
    // class Boostrap{

    //};//class bootstrap

  }  // namespace stat
}  // namespace cytnx

#endif
