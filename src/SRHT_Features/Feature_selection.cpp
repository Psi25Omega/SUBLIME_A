// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>

using namespace Rcpp;
using namespace Eigen;

// --- Helper Functions ---

// 1. Iterative Fast Walsh-Hadamard Transform
// (Faster than recursive, works on contiguous Row-Major memory)
void fwht_iterative(double* a, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += 2 * len) {
            for (int j = 0; j < len; j++) {
                double u = a[i + j];
                double v = a[i + len + j];
                a[i + j] = u + v;
                a[i + len + j] = u - v;
            }
        }
    }
}

// 2. Parallel Scaling to Range [-1, 1]
// (Prevents Numerical Overflow in Leverage Method)
void scaleData(Eigen::MatrixXd& X) {
    double max_val = 0.0;
    int n = X.rows();
    int d = X.cols();

    // Parallel Max Reduction
    #pragma omp parallel for reduction(max:max_val)
    for(int i=0; i<n; ++i) {
        for(int j=0; j<d; ++j) {
            double abs_v = std::abs(X(i,j));
            if(std::isfinite(abs_v) && abs_v > max_val) max_val = abs_v;
        }
    }
    
    // Parallel Scaling
    if(max_val > 1e-9) {
        double scale = 1.0 / max_val;
        #pragma omp parallel for
        for(int i=0; i<n; ++i) {
            for(int j=0; j<d; ++j) {
                if(std::isfinite(X(i,j))) X(i,j) *= scale;
                else X(i,j) = 0.0;
            }
        }
    }
}

int nextPowerOfTwo(int n) {
    if ((n > 0) && ((n & (n - 1)) == 0)) return n;
    return std::pow(2, std::ceil(std::log2(n)));
}

// Binning for Supervised Method
std::vector<int> bin_continuous_targets(const Eigen::VectorXd& y, int n_bins) {
    int n = y.size();
    std::vector<int> labels(n);
    double min_y = y.minCoeff();
    double max_y = y.maxCoeff();
    double range = max_y - min_y;
    if (range < 1e-9 || !std::isfinite(range)) return std::vector<int>(n, 0);
    
    for(int i = 0; i < n; i++) {
        if (!std::isfinite(y[i])) { labels[i] = 0; continue; }
        int bin = (int)((y[i] - min_y) / range * n_bins);
        if (bin >= n_bins) bin = n_bins - 1;
        labels[i] = bin;
    }
    return labels;
}

struct OLSResult {
    double r2;
    Eigen::VectorXd coeffs;
};

// Robust OLS Solver
OLSResult solve_ols(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    int n = X.rows();
    int k = X.cols();
    if (n == 0 || k == 0) return {0.0, Eigen::VectorXd::Zero(k+1)};

    Eigen::MatrixXd X_bias(n, k + 1);
    X_bias.col(0) = Eigen::VectorXd::Ones(n); // Intercept
    X_bias.block(0, 1, n, k) = X;
    
    // Use ColPivHouseholderQr for stability
    Eigen::VectorXd w = X_bias.colPivHouseholderQr().solve(y);
    
    if (!w.allFinite()) return {0.0, Eigen::VectorXd::Zero(k+1)};

    Eigen::VectorXd y_pred = X_bias * w;
    double y_mean = y.mean();
    double ss_tot = (y.array() - y_mean).square().sum();
    double ss_res = (y.array() - y_pred.array()).square().sum();
    
    if (ss_tot < 1e-9) return {0.0, w};
    double r2 = 1.0 - (ss_res / ss_tot);
    
    return {r2, w};
}

// --- Core Logic ---

class ISRHT_Core {
public:
    static Eigen::MatrixXd rotateData(Eigen::MatrixXd X, int seed) {
        // 1. Parallel Scaling
        scaleData(X);

        std::mt19937 rng(seed);
        int n = X.rows();
        int d = X.cols();
        int padded_d = nextPowerOfTwo(d);
        
        // 2. Generate Global Signs (Diagonal D)
        // Must be generated once and applied to all rows identically
        std::uniform_int_distribution<> dist(0, 1);
        std::vector<double> signs(padded_d);
        for(int j=0; j<padded_d; ++j) {
            signs[j] = (dist(rng) == 0) ? 1.0 : -1.0;
        }

        // 3. OPTIMIZATION: Use Row-Major Layout
        // This makes rows contiguous in memory, allowing vectorization 
        // and correct pointer access for the Hadamard transform.
        using MatrixRowMaj = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        MatrixRowMaj X_rotated = MatrixRowMaj::Zero(n, padded_d);
        
        // Copy input block
        X_rotated.block(0, 0, n, d) = X;
        
        double scale = 1.0 / std::sqrt((double)padded_d);
        
        // 4. Data Parallel Rotation
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            // Apply Signs
            for (int j = 0; j < d; j++) {
                X_rotated(i, j) *= signs[j];
            }
            // Apply Iterative FWHT (Fast on contiguous memory)
            fwht_iterative(X_rotated.row(i).data(), padded_d);
        }
        
        X_rotated *= scale;

        // Implicitly converts back to Col-Major for standard Eigen operations
        return X_rotated; 
    }

    // --- Sampling Methods ---

    static std::pair<Eigen::MatrixXd, std::vector<int>> fit_transform_uniform(const Eigen::MatrixXd& X_rot, int r, std::mt19937& rng) {
        int n = X_rot.rows();
        int d = X_rot.cols();
        std::vector<int> indices(d);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        indices.resize(r); 

        Eigen::MatrixXd X_new(n, r);
        double scale = std::sqrt((double)d / r);
        for (int j = 0; j < r; j++) X_new.col(j) = X_rot.col(indices[j]) * scale;
        return {X_new, indices};
    }

    static std::pair<Eigen::MatrixXd, std::vector<int>> fit_transform_top_r(const Eigen::MatrixXd& X_rot, int r) {
        int d = X_rot.cols();
        Eigen::VectorXd norms = X_rot.colwise().squaredNorm();
        std::vector<std::pair<double, int>> col_norms(d);
        for (int i = 0; i < d; i++) {
            double val = std::isfinite(norms[i]) ? norms[i] : -1.0;
            col_norms[i] = {val, i};
        }
        std::sort(col_norms.rbegin(), col_norms.rend());

        std::vector<int> indices(r);
        Eigen::MatrixXd X_new(X_rot.rows(), r);
        for (int j = 0; j < r; j++) {
            indices[j] = col_norms[j].second;
            X_new.col(j) = X_rot.col(indices[j]);
        }
        return {X_new, indices};
    }

    static std::pair<Eigen::MatrixXd, std::vector<int>> fit_transform_leverage(const Eigen::MatrixXd& X_rot, int r, std::mt19937& rng) {
        int n = X_rot.rows();
        int d = X_rot.cols();
        Eigen::VectorXd norms = X_rot.colwise().squaredNorm();
        
        std::vector<double> weights(d);
        double max_norm = 0.0;
        
        // Find max for safe scaling
        for(int i=0; i<d; ++i) {
            if(std::isfinite(norms[i])) {
                if(norms[i] > max_norm) max_norm = norms[i];
            }
        }

        if(max_norm > 1e-100) { 
            for(int i=0; i<d; ++i) {
                if(std::isfinite(norms[i]) && norms[i] > 0) weights[i] = norms[i] / max_norm; 
                else weights[i] = 0.0;
            }
        } else {
            std::fill(weights.begin(), weights.end(), 1.0);
        }

        std::discrete_distribution<> dist(weights.begin(), weights.end());
        std::vector<int> indices;
        std::vector<bool> is_selected(d, false);
        int count = 0, attempts = 0;
        while (count < r && attempts < r*50) {
            int idx = dist(rng);
            if (!is_selected[idx]) {
                is_selected[idx] = true;
                indices.push_back(idx);
                count++;
            }
            attempts++;
        }
        for(int i=0; i<d && count < r; ++i) {
            if(!is_selected[i]) { indices.push_back(i); count++; }
        }

        Eigen::MatrixXd X_new(n, r);
        for (int j = 0; j < r; j++) X_new.col(j) = X_rot.col(indices[j]);
        return {X_new, indices};
    }

    static std::pair<Eigen::MatrixXd, std::vector<int>> fit_transform_supervised(const Eigen::MatrixXd& X_rot, const std::vector<int>& labels, int r, double a_param) {
        int n = X_rot.rows();
        int d = X_rot.cols();
        int max_label = *std::max_element(labels.begin(), labels.end());
        int num_classes = max_label + 1;
        
        Eigen::MatrixXd class_sums = Eigen::MatrixXd::Zero(num_classes, d);
        Eigen::MatrixXd class_sq_sums = Eigen::MatrixXd::Zero(num_classes, d);
        std::vector<int> class_counts(num_classes, 0);
        
        for(int i=0; i<n; ++i) {
            int c = labels[i];
            class_counts[c]++;
            class_sums.row(c) += X_rot.row(i);
            class_sq_sums.row(c) += X_rot.row(i).array().square().matrix();
        }
        
        Eigen::VectorXd total_sums = X_rot.colwise().sum(); 
        Eigen::VectorXd term_Av = Eigen::VectorXd::Zero(d);
        Eigen::VectorXd term_Dv = Eigen::VectorXd::Zero(d);
        
        for (int c = 0; c < num_classes; ++c) {
            if (class_counts[c] == 0) continue;
            term_Av += (class_sums.row(c).array().square()).transpose().matrix();
            double coeff = (double)class_counts[c] - a_param * (n - (double)class_counts[c]);
            term_Dv += coeff * class_sq_sums.row(c).transpose();
        }
        term_Av *= (1.0 + a_param);
        term_Av -= a_param * (total_sums.array().square()).transpose().matrix();
        
        Eigen::VectorXd b_scores = term_Dv - term_Av;
        
        std::vector<std::pair<double, int>> sorted_scores(d);
        for(int j=0; j<d; ++j) {
            double s = b_scores[j];
            if (!std::isfinite(s)) s = 1e20; 
            sorted_scores[j] = {s, j};
        }
        std::sort(sorted_scores.begin(), sorted_scores.end());
        
        std::vector<int> indices(r);
        Eigen::MatrixXd X_new(n, r);
        for (int j = 0; j < r; j++) {
            indices[j] = sorted_scores[j].second;
            X_new.col(j) = X_rot.col(indices[j]);
        }
        return {X_new, indices};
    }
};

struct BenchmarkResult {
    std::string method;
    double r2;
    Eigen::VectorXd coeffs;
    std::vector<int> indices;
    int total_features;
};

// [[Rcpp::export]]
List run_isrht_benchmark(Eigen::MatrixXd X, Eigen::VectorXd y, int r, int bins) {
    int seed = 123;
    Eigen::MatrixXd X_rot = ISRHT_Core::rotateData(X, seed);
    int total_d = X_rot.cols();
    std::vector<int> labels = bin_continuous_targets(y, bins);
    
    std::vector<BenchmarkResult> all_results(4);
    
    // Parallel Sections: Run all 4 methods concurrently
    #pragma omp parallel sections
    {
        #pragma omp section 
        {
            std::mt19937 thread_rng(seed + 1);
            auto pair = ISRHT_Core::fit_transform_uniform(X_rot, r, thread_rng);
            OLSResult res = solve_ols(pair.first, y);
            all_results[0] = {"Uniform", res.r2, res.coeffs, pair.second, total_d};
        }
        #pragma omp section
        {
            auto pair = ISRHT_Core::fit_transform_top_r(X_rot, r);
            OLSResult res = solve_ols(pair.first, y);
            all_results[1] = {"Top-r", res.r2, res.coeffs, pair.second, total_d};
        }
        #pragma omp section
        {
            std::mt19937 thread_rng(seed + 2);
            auto pair = ISRHT_Core::fit_transform_leverage(X_rot, r, thread_rng);
            OLSResult res = solve_ols(pair.first, y);
            all_results[2] = {"Leverage", res.r2, res.coeffs, pair.second, total_d};
        }
        #pragma omp section
        {
            auto pair = ISRHT_Core::fit_transform_supervised(X_rot, labels, r, 1.0);
            OLSResult res = solve_ols(pair.first, y);
            all_results[3] = {"Supervised", res.r2, res.coeffs, pair.second, total_d};
        }
    }
    
    List output(4);
    for(int i=0; i<4; ++i) {
        output[i] = List::create(
            Named("Method") = all_results[i].method,
            Named("R_Squared") = all_results[i].r2,
            Named("Coefficients") = all_results[i].coeffs,
            Named("Indices") = all_results[i].indices,
            Named("TotalFeatures") = all_results[i].total_features
        );
    }
    return output;
}
