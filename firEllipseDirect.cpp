#include "opencv2\core\core.hpp"
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2\core/core.hpp"
#include "opencv2/core/base.hpp"
#include "fitEllipse.h"
#include <iostream>

#pragma warning(disable:4996)
#pragma warning(disable:4244)


template<typename _Tp> static bool
isSymmetric_(cv::InputArray src) {
	cv::Mat _src = src.getMat();
	if (_src.cols != _src.rows)
		return false;
	for (int i = 0; i < _src.rows; i++) {
		for (int j = 0; j < _src.cols; j++) {
			_Tp a = _src.at<_Tp>(i, j);
			_Tp b = _src.at<_Tp>(j, i);
			if (a != b) {
				return false;
			}
		}
	}
	return true;
}

template<typename _Tp> static bool
isSymmetric_(cv::InputArray src, double eps) {
	cv::Mat _src = src.getMat();
	if (_src.cols != _src.rows)
		return false;
	for (int i = 0; i < _src.rows; i++) {
		for (int j = 0; j < _src.cols; j++) {
			_Tp a = _src.at<_Tp>(i, j);
			_Tp b = _src.at<_Tp>(j, i);
			if (std::abs(a - b) > eps) {
				return false;
			}
		}
	}
	return true;
}

static bool isSymmetric(cv::InputArray src, double eps = 1e-16)
{
	cv::Mat m = src.getMat();
	switch (m.type()) {
	case CV_8SC1: return isSymmetric_<char>(m); break;
	case CV_8UC1:
		return isSymmetric_<unsigned char>(m); break;
	case CV_16SC1:
		return isSymmetric_<short>(m); break;
	case CV_16UC1:
		return isSymmetric_<unsigned short>(m); break;
	case CV_32SC1:
		return isSymmetric_<int>(m); break;
	case CV_32FC1:
		return isSymmetric_<float>(m, eps); break;
	case CV_64FC1:
		return isSymmetric_<double>(m, eps); break;
	default:
		break;
	}
	return false;
}

class EigenvalueDecomposition {
private:

	// Holds the data dimension.
	int n;

	// Stores real/imag part of a complex division.
	double cdivr, cdivi;

	// Pointer to internal memory.
	double *d, *e, *ort;
	double **V, **H;

	// Holds the computed eigenvalues.
	cv::Mat _eigenvalues;

	// Holds the computed eigenvectors.
	cv::Mat _eigenvectors;

	// Allocates memory.
	template<typename _Tp>
	_Tp *alloc_1d(int m) {
		return new _Tp[m];
	}

	// Allocates memory.
	template<typename _Tp>
	_Tp *alloc_1d(int m, _Tp val) {
		_Tp *arr = alloc_1d<_Tp>(m);
		for (int i = 0; i < m; i++)
			arr[i] = val;
		return arr;
	}

	// Allocates memory.
	template<typename _Tp>
	_Tp **alloc_2d(int m, int _n) {
		_Tp **arr = new _Tp*[m];
		for (int i = 0; i < m; i++)
			arr[i] = new _Tp[_n];
		return arr;
	}

	// Allocates memory.
	template<typename _Tp>
	_Tp **alloc_2d(int m, int _n, _Tp val) {
		_Tp **arr = alloc_2d<_Tp>(m, _n);
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < _n; j++) {
				arr[i][j] = val;
			}
		}
		return arr;
	}

	void cdiv(double xr, double xi, double yr, double yi) {
		double r, dv;
		if (std::abs(yr) > std::abs(yi)) {
			r = yi / yr;
			dv = yr + r * yi;
			cdivr = (xr + r * xi) / dv;
			cdivi = (xi - r * xr) / dv;
		}
		else {
			r = yr / yi;
			dv = yi + r * yr;
			cdivr = (r * xr + xi) / dv;
			cdivi = (r * xi - xr) / dv;
		}
	}

	// Nonsymmetric reduction from Hessenberg to real Schur form.

	void hqr2() {

		//  This is derived from the Algol procedure hqr2,
		//  by Martin and Wilkinson, Handbook for Auto. Comp.,
		//  Vol.ii-Linear Algebra, and the corresponding
		//  Fortran subroutine in EISPACK.

		// Initialize
		int nn = this->n;
		int n1 = nn - 1;
		int low = 0;
		int high = nn - 1;
		double eps = std::pow(2.0, -52.0);
		double exshift = 0.0;
		double p = 0, q = 0, r = 0, s = 0, z = 0, t, w, x, y;

		// Store roots isolated by balanc and compute matrix norm

		double norm = 0.0;
		for (int i = 0; i < nn; i++) {
			if (i < low || i > high) {
				d[i] = H[i][i];
				e[i] = 0.0;
			}
			for (int j = std::max(i - 1, 0); j < nn; j++) {
				norm = norm + std::abs(H[i][j]);
			}
		}

		// Outer loop over eigenvalue index
		int iter = 0;
		while (n1 >= low) {

			// Look for single small sub-diagonal element
			int l = n1;
			while (l > low) {
				s = std::abs(H[l - 1][l - 1]) + std::abs(H[l][l]);
				if (s == 0.0) {
					s = norm;
				}
				if (std::abs(H[l][l - 1]) < eps * s) {
					break;
				}
				l--;
			}

			// Check for convergence
			// One root found

			if (l == n1) {
				H[n1][n1] = H[n1][n1] + exshift;
				d[n1] = H[n1][n1];
				e[n1] = 0.0;
				n1--;
				iter = 0;

				// Two roots found

			}
			else if (l == n1 - 1) {
				w = H[n1][n1 - 1] * H[n1 - 1][n1];
				p = (H[n1 - 1][n1 - 1] - H[n1][n1]) / 2.0;
				q = p * p + w;
				z = std::sqrt(std::abs(q));
				H[n1][n1] = H[n1][n1] + exshift;
				H[n1 - 1][n1 - 1] = H[n1 - 1][n1 - 1] + exshift;
				x = H[n1][n1];

				// Real pair

				if (q >= 0) {
					if (p >= 0) {
						z = p + z;
					}
					else {
						z = p - z;
					}
					d[n1 - 1] = x + z;
					d[n1] = d[n1 - 1];
					if (z != 0.0) {
						d[n1] = x - w / z;
					}
					e[n1 - 1] = 0.0;
					e[n1] = 0.0;
					x = H[n1][n1 - 1];
					s = std::abs(x) + std::abs(z);
					p = x / s;
					q = z / s;
					r = std::sqrt(p * p + q * q);
					p = p / r;
					q = q / r;

					// Row modification

					for (int j = n1 - 1; j < nn; j++) {
						z = H[n1 - 1][j];
						H[n1 - 1][j] = q * z + p * H[n1][j];
						H[n1][j] = q * H[n1][j] - p * z;
					}

					// Column modification

					for (int i = 0; i <= n1; i++) {
						z = H[i][n1 - 1];
						H[i][n1 - 1] = q * z + p * H[i][n1];
						H[i][n1] = q * H[i][n1] - p * z;
					}

					// Accumulate transformations

					for (int i = low; i <= high; i++) {
						z = V[i][n1 - 1];
						V[i][n1 - 1] = q * z + p * V[i][n1];
						V[i][n1] = q * V[i][n1] - p * z;
					}

					// Complex pair

				}
				else {
					d[n1 - 1] = x + p;
					d[n1] = x + p;
					e[n1 - 1] = z;
					e[n1] = -z;
				}
				n1 = n1 - 2;
				iter = 0;

				// No convergence yet

			}
			else {

				// Form shift

				x = H[n1][n1];
				y = 0.0;
				w = 0.0;
				if (l < n1) {
					y = H[n1 - 1][n1 - 1];
					w = H[n1][n1 - 1] * H[n1 - 1][n1];
				}

				// Wilkinson's original ad hoc shift

				if (iter == 10) {
					exshift += x;
					for (int i = low; i <= n1; i++) {
						H[i][i] -= x;
					}
					s = std::abs(H[n1][n1 - 1]) + std::abs(H[n1 - 1][n1 - 2]);
					x = y = 0.75 * s;
					w = -0.4375 * s * s;
				}

				// MATLAB's new ad hoc shift

				if (iter == 30) {
					s = (y - x) / 2.0;
					s = s * s + w;
					if (s > 0) {
						s = std::sqrt(s);
						if (y < x) {
							s = -s;
						}
						s = x - w / ((y - x) / 2.0 + s);
						for (int i = low; i <= n1; i++) {
							H[i][i] -= s;
						}
						exshift += s;
						x = y = w = 0.964;
					}
				}

				iter = iter + 1; // (Could check iteration count here.)

								 // Look for two consecutive small sub-diagonal elements
				int m = n1 - 2;
				while (m >= l) {
					z = H[m][m];
					r = x - z;
					s = y - z;
					p = (r * s - w) / H[m + 1][m] + H[m][m + 1];
					q = H[m + 1][m + 1] - z - r - s;
					r = H[m + 2][m + 1];
					s = std::abs(p) + std::abs(q) + std::abs(r);
					p = p / s;
					q = q / s;
					r = r / s;
					if (m == l) {
						break;
					}
					if (std::abs(H[m][m - 1]) * (std::abs(q) + std::abs(r)) < eps * (std::abs(p)
						* (std::abs(H[m - 1][m - 1]) + std::abs(z) + std::abs(
							H[m + 1][m + 1])))) {
						break;
					}
					m--;
				}

				for (int i = m + 2; i <= n1; i++) {
					H[i][i - 2] = 0.0;
					if (i > m + 2) {
						H[i][i - 3] = 0.0;
					}
				}

				// Double QR step involving rows l:n and columns m:n

				for (int k = m; k < n1; k++) {
					bool notlast = (k != n1 - 1);
					if (k != m) {
						p = H[k][k - 1];
						q = H[k + 1][k - 1];
						r = (notlast ? H[k + 2][k - 1] : 0.0);
						x = std::abs(p) + std::abs(q) + std::abs(r);
						if (x != 0.0) {
							p = p / x;
							q = q / x;
							r = r / x;
						}
					}
					if (x == 0.0) {
						break;
					}
					s = std::sqrt(p * p + q * q + r * r);
					if (p < 0) {
						s = -s;
					}
					if (s != 0) {
						if (k != m) {
							H[k][k - 1] = -s * x;
						}
						else if (l != m) {
							H[k][k - 1] = -H[k][k - 1];
						}
						p = p + s;
						x = p / s;
						y = q / s;
						z = r / s;
						q = q / p;
						r = r / p;

						// Row modification

						for (int j = k; j < nn; j++) {
							p = H[k][j] + q * H[k + 1][j];
							if (notlast) {
								p = p + r * H[k + 2][j];
								H[k + 2][j] = H[k + 2][j] - p * z;
							}
							H[k][j] = H[k][j] - p * x;
							H[k + 1][j] = H[k + 1][j] - p * y;
						}

						// Column modification

						for (int i = 0; i <= std::min(n1, k + 3); i++) {
							p = x * H[i][k] + y * H[i][k + 1];
							if (notlast) {
								p = p + z * H[i][k + 2];
								H[i][k + 2] = H[i][k + 2] - p * r;
							}
							H[i][k] = H[i][k] - p;
							H[i][k + 1] = H[i][k + 1] - p * q;
						}

						// Accumulate transformations

						for (int i = low; i <= high; i++) {
							p = x * V[i][k] + y * V[i][k + 1];
							if (notlast) {
								p = p + z * V[i][k + 2];
								V[i][k + 2] = V[i][k + 2] - p * r;
							}
							V[i][k] = V[i][k] - p;
							V[i][k + 1] = V[i][k + 1] - p * q;
						}
					} // (s != 0)
				} // k loop
			} // check convergence
		} // while (n1 >= low)

		  // Backsubstitute to find vectors of upper triangular form

		if (norm == 0.0) {
			return;
		}

		for (n1 = nn - 1; n1 >= 0; n1--) {
			p = d[n1];
			q = e[n1];

			// Real vector

			if (q == 0) {
				int l = n1;
				H[n1][n1] = 1.0;
				for (int i = n1 - 1; i >= 0; i--) {
					w = H[i][i] - p;
					r = 0.0;
					for (int j = l; j <= n1; j++) {
						r = r + H[i][j] * H[j][n1];
					}
					if (e[i] < 0.0) {
						z = w;
						s = r;
					}
					else {
						l = i;
						if (e[i] == 0.0) {
							if (w != 0.0) {
								H[i][n1] = -r / w;
							}
							else {
								H[i][n1] = -r / (eps * norm);
							}

							// Solve real equations

						}
						else {
							x = H[i][i + 1];
							y = H[i + 1][i];
							q = (d[i] - p) * (d[i] - p) + e[i] * e[i];
							t = (x * s - z * r) / q;
							H[i][n1] = t;
							if (std::abs(x) > std::abs(z)) {
								H[i + 1][n1] = (-r - w * t) / x;
							}
							else {
								H[i + 1][n1] = (-s - y * t) / z;
							}
						}

						// Overflow control

						t = std::abs(H[i][n1]);
						if ((eps * t) * t > 1) {
							for (int j = i; j <= n1; j++) {
								H[j][n1] = H[j][n1] / t;
							}
						}
					}
				}
				// Complex vector
			}
			else if (q < 0) {
				int l = n1 - 1;

				// Last vector component imaginary so matrix is triangular

				if (std::abs(H[n1][n1 - 1]) > std::abs(H[n1 - 1][n1])) {
					H[n1 - 1][n1 - 1] = q / H[n1][n1 - 1];
					H[n1 - 1][n1] = -(H[n1][n1] - p) / H[n1][n1 - 1];
				}
				else {
					cdiv(0.0, -H[n1 - 1][n1], H[n1 - 1][n1 - 1] - p, q);
					H[n1 - 1][n1 - 1] = cdivr;
					H[n1 - 1][n1] = cdivi;
				}
				H[n1][n1 - 1] = 0.0;
				H[n1][n1] = 1.0;
				for (int i = n1 - 2; i >= 0; i--) {
					double ra, sa, vr, vi;
					ra = 0.0;
					sa = 0.0;
					for (int j = l; j <= n1; j++) {
						ra = ra + H[i][j] * H[j][n1 - 1];
						sa = sa + H[i][j] * H[j][n1];
					}
					w = H[i][i] - p;

					if (e[i] < 0.0) {
						z = w;
						r = ra;
						s = sa;
					}
					else {
						l = i;
						if (e[i] == 0) {
							cdiv(-ra, -sa, w, q);
							H[i][n1 - 1] = cdivr;
							H[i][n1] = cdivi;
						}
						else {

							// Solve complex equations

							x = H[i][i + 1];
							y = H[i + 1][i];
							vr = (d[i] - p) * (d[i] - p) + e[i] * e[i] - q * q;
							vi = (d[i] - p) * 2.0 * q;
							if (vr == 0.0 && vi == 0.0) {
								vr = eps * norm * (std::abs(w) + std::abs(q) + std::abs(x)
									+ std::abs(y) + std::abs(z));
							}
							cdiv(x * r - z * ra + q * sa,
								x * s - z * sa - q * ra, vr, vi);
							H[i][n1 - 1] = cdivr;
							H[i][n1] = cdivi;
							if (std::abs(x) > (std::abs(z) + std::abs(q))) {
								H[i + 1][n1 - 1] = (-ra - w * H[i][n1 - 1] + q
									* H[i][n1]) / x;
								H[i + 1][n1] = (-sa - w * H[i][n1] - q * H[i][n1
									- 1]) / x;
							}
							else {
								cdiv(-r - y * H[i][n1 - 1], -s - y * H[i][n1], z,
									q);
								H[i + 1][n1 - 1] = cdivr;
								H[i + 1][n1] = cdivi;
							}
						}

						// Overflow control

						t = std::max(std::abs(H[i][n1 - 1]), std::abs(H[i][n1]));
						if ((eps * t) * t > 1) {
							for (int j = i; j <= n1; j++) {
								H[j][n1 - 1] = H[j][n1 - 1] / t;
								H[j][n1] = H[j][n1] / t;
							}
						}
					}
				}
			}
		}

		// Vectors of isolated roots

		for (int i = 0; i < nn; i++) {
			if (i < low || i > high) {
				for (int j = i; j < nn; j++) {
					V[i][j] = H[i][j];
				}
			}
		}

		// Back transformation to get eigenvectors of original matrix

		for (int j = nn - 1; j >= low; j--) {
			for (int i = low; i <= high; i++) {
				z = 0.0;
				for (int k = low; k <= std::min(j, high); k++) {
					z = z + V[i][k] * H[k][j];
				}
				V[i][j] = z;
			}
		}
	}

	// Nonsymmetric reduction to Hessenberg form.
	void orthes() {
		//  This is derived from the Algol procedures orthes and ortran,
		//  by Martin and Wilkinson, Handbook for Auto. Comp.,
		//  Vol.ii-Linear Algebra, and the corresponding
		//  Fortran subroutines in EISPACK.
		int low = 0;
		int high = n - 1;

		for (int m = low + 1; m < high; m++) {

			// Scale column.

			double scale = 0.0;
			for (int i = m; i <= high; i++) {
				scale = scale + std::abs(H[i][m - 1]);
			}
			if (scale != 0.0) {

				// Compute Householder transformation.

				double h = 0.0;
				for (int i = high; i >= m; i--) {
					ort[i] = H[i][m - 1] / scale;
					h += ort[i] * ort[i];
				}
				double g = std::sqrt(h);
				if (ort[m] > 0) {
					g = -g;
				}
				h = h - ort[m] * g;
				ort[m] = ort[m] - g;

				// Apply Householder similarity transformation
				// H = (I-u*u'/h)*H*(I-u*u')/h)

				for (int j = m; j < n; j++) {
					double f = 0.0;
					for (int i = high; i >= m; i--) {
						f += ort[i] * H[i][j];
					}
					f = f / h;
					for (int i = m; i <= high; i++) {
						H[i][j] -= f * ort[i];
					}
				}

				for (int i = 0; i <= high; i++) {
					double f = 0.0;
					for (int j = high; j >= m; j--) {
						f += ort[j] * H[i][j];
					}
					f = f / h;
					for (int j = m; j <= high; j++) {
						H[i][j] -= f * ort[j];
					}
				}
				ort[m] = scale * ort[m];
				H[m][m - 1] = scale * g;
			}
		}

		// Accumulate transformations (Algol's ortran).

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				V[i][j] = (i == j ? 1.0 : 0.0);
			}
		}

		for (int m = high - 1; m > low; m--) {
			if (H[m][m - 1] != 0.0) {
				for (int i = m + 1; i <= high; i++) {
					ort[i] = H[i][m - 1];
				}
				for (int j = m; j <= high; j++) {
					double g = 0.0;
					for (int i = m; i <= high; i++) {
						g += ort[i] * V[i][j];
					}
					// Double division avoids possible underflow
					g = (g / ort[m]) / H[m][m - 1];
					for (int i = m; i <= high; i++) {
						V[i][j] += g * ort[i];
					}
				}
			}
		}
	}

	// Releases all internal working memory.
	void release() {
		// releases the working data
		delete[] d;
		delete[] e;
		delete[] ort;
		for (int i = 0; i < n; i++) {
			delete[] H[i];
			delete[] V[i];
		}
		delete[] H;
		delete[] V;
	}

	// Computes the Eigenvalue Decomposition for a matrix given in H.
	void compute() {
		// Allocate memory for the working data.
		V = alloc_2d<double>(n, n, 0.0);
		d = alloc_1d<double>(n);
		e = alloc_1d<double>(n);
		ort = alloc_1d<double>(n);
		//try {
		//CV_TRY{
		// Reduce to Hessenberg form.
		orthes();
		// Reduce Hessenberg to real Schur form.
		hqr2();
		// Copy eigenvalues to OpenCV Matrix.
		_eigenvalues.create(1, n, CV_64FC1);
		for (int i = 0; i < n; i++) {
			_eigenvalues.at<double>(0, i) = d[i];
		}
		// Copy eigenvectors to OpenCV Matrix.
		_eigenvectors.create(n, n, CV_64FC1);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++)
				_eigenvectors.at<double>(i, j) = V[i][j];
		// Deallocate the memory by releasing all internal working data.
		release();
		//}
		//catch	//CV_CATCH_ALL
		{
			//release();
			//CV_RETHROW();
		}
	}

public:
	// Initializes & computes the Eigenvalue Decomposition for a general matrix
	// given in src. This function is a port of the EigenvalueSolver in JAMA,
	// which has been released to public domain by The MathWorks and the
	// National Institute of Standards and Technology (NIST).
	EigenvalueDecomposition(cv::InputArray src, bool fallbackSymmetric = true) {
		compute(src, fallbackSymmetric);
	}

	// This function computes the Eigenvalue Decomposition for a general matrix
	// given in src. This function is a port of the EigenvalueSolver in JAMA,
	// which has been released to public domain by The MathWorks and the
	// National Institute of Standards and Technology (NIST).
	void compute(cv::InputArray src, bool fallbackSymmetric)
	{
		//CV_INSTRUMENT_REGION()

		if (fallbackSymmetric && isSymmetric(src)) {
			// Fall back to OpenCV for a symmetric matrix!
			cv::eigen(src, _eigenvalues, _eigenvectors);
		}
		else {
			cv::Mat tmp;
			// Convert the given input matrix to double. Is there any way to
			// prevent allocating the temporary memory? Only used for copying
			// into working memory and deallocated after.
			src.getMat().convertTo(tmp, CV_64FC1);
			// Get dimension of the matrix.
			this->n = tmp.cols;
			// Allocate the matrix data to work on.
			this->H = alloc_2d<double>(n, n);
			// Now safely copy the data.
			for (int i = 0; i < tmp.rows; i++) {
				for (int j = 0; j < tmp.cols; j++) {
					this->H[i][j] = tmp.at<double>(i, j);
				}
			}
			// Deallocates the temporary matrix before computing.
			tmp.release();
			// Performs the eigenvalue decomposition of H.
			compute();
		}
	}

	~EigenvalueDecomposition() {}

	// Returns the eigenvalues of the Eigenvalue Decomposition.
	cv::Mat eigenvalues() const { return _eigenvalues; }
	// Returns the eigenvectors of the Eigenvalue Decomposition.
	cv::Mat eigenvectors() const { return _eigenvectors; }
};

enum SortFlags {
	SORT_EVERY_ROW = 0, //!< each matrix row is sorted independently
	SORT_EVERY_COLUMN = 1, //!< each matrix column is sorted
						   //!< independently; this flag and the previous one are
						   //!< mutually exclusive.
						   SORT_ASCENDING = 0, //!< each matrix row is sorted in the ascending
											   //!< order.
											   SORT_DESCENDING = 16 //!< each matrix row is sorted in the
																	//!< descending order; this flag and the previous one are also
																	//!< mutually exclusive.
};

void eigenNonSymmetric(cv::InputArray _src, cv::OutputArray _evals, cv::OutputArray _evects)
{
	//CV_INSTRUMENT_REGION()

	cv::Mat src = _src.getMat();
	int type = src.type();
	size_t n = (size_t)src.rows;

	CV_Assert(src.rows == src.cols);
	CV_Assert(type == CV_32F || type == CV_64F);

	cv::Mat src64f;
	if (type == CV_32F)
		src.convertTo(src64f, CV_32FC1);
	else
		src64f = src;

	EigenvalueDecomposition eigensystem(src64f, false);

	// EigenvalueDecomposition returns transposed and non-sorted eigenvalues
	std::vector<double> eigenvalues64f;
	eigensystem.eigenvalues().copyTo(eigenvalues64f);
	CV_Assert(eigenvalues64f.size() == n);

	std::vector<int> sort_indexes(n);
	cv::sortIdx(eigenvalues64f, sort_indexes, SORT_EVERY_ROW | SORT_DESCENDING);

	std::vector<double> sorted_eigenvalues64f(n);
	for (size_t i = 0; i < n; i++) sorted_eigenvalues64f[i] = eigenvalues64f[sort_indexes[i]];

	cv::Mat(sorted_eigenvalues64f).convertTo(_evals, type);

	if (_evects.needed())
	{
		cv::Mat eigenvectors64f = eigensystem.eigenvectors().t(); // transpose
		CV_Assert((size_t)eigenvectors64f.rows == n);
		CV_Assert((size_t)eigenvectors64f.cols == n);
		cv::Mat_<double> sorted_eigenvectors64f((int)n, (int)n, CV_64FC1);
		for (size_t i = 0; i < n; i++)
		{
			double* pDst = sorted_eigenvectors64f.ptr<double>((int)i);
			double* pSrc = eigenvectors64f.ptr<double>(sort_indexes[(int)i]);
			CV_Assert(pSrc != NULL);
			memcpy(pDst, pSrc, n * sizeof(double));
		}
		sorted_eigenvectors64f.convertTo(_evects, type);
	}
}

cv::RotatedRect fitEllipseDirect(cv::InputArray _points, cv::Mat &rawSol)
{
	cv::Mat points = _points.getMat();
	int i, n = points.checkVector(2);
	int depth = points.depth();
	CV_Assert(n >= 0 && (depth == CV_32F || depth == CV_32S));

	cv::RotatedRect box;

	if (n < 5)
		CV_Error(CV_StsBadSize, "There should be at least 5 points to fit the ellipse");

	cv::Point2f c(0, 0);

	bool is_float = (depth == CV_32F);
	const cv::Point*   ptsi = points.ptr<cv::Point>();
	const cv::Point2f* ptsf = points.ptr<cv::Point2f>();

	cv::Mat A(n, 6, CV_64F);
	cv::Matx<double, 6, 6> DM;
	cv::Matx33d M, TM, Q;
	cv::Matx<double, 3, 1> pVec;

	double x0, y0, a, b, theta, Ts;

	//for (i = 0; i < n; i++)
	//{
	//	cv::Point2f p = is_float ? ptsf[i] : cv::Point2f((float)ptsi[i].x, (float)ptsi[i].y);
	//	c += p;
	//}
	//c.x /= (float)n;
	//c.y /= (float)n;

	for (i = 0; i < n; i++)
	{
		cv::Point2f p = is_float ? ptsf[i] : cv::Point2f((float)ptsi[i].x, (float)ptsi[i].y);
		p -= c;

		A.at<double>(i, 0) = (double)(p.x)*(p.x);
		A.at<double>(i, 1) = (double)(p.x)*(p.y);
		A.at<double>(i, 2) = (double)(p.y)*(p.y);
		A.at<double>(i, 3) = (double)p.x;
		A.at<double>(i, 4) = (double)p.y;
		A.at<double>(i, 5) = 1.0;
	}
	cv::mulTransposed(A, DM, true, cv::noArray(), 1.0, -1);
	DM *= (1.0 / n);

	TM(0, 0) = DM(0, 5)*DM(3, 5)*DM(4, 4) - DM(0, 5)*DM(3, 4)*DM(4, 5) - DM(0, 4)*DM(3, 5)*DM(5, 4) + \
		DM(0, 3)*DM(4, 5)*DM(5, 4) + DM(0, 4)*DM(3, 4)*DM(5, 5) - DM(0, 3)*DM(4, 4)*DM(5, 5);
	TM(0, 1) = DM(1, 5)*DM(3, 5)*DM(4, 4) - DM(1, 5)*DM(3, 4)*DM(4, 5) - DM(1, 4)*DM(3, 5)*DM(5, 4) + \
		DM(1, 3)*DM(4, 5)*DM(5, 4) + DM(1, 4)*DM(3, 4)*DM(5, 5) - DM(1, 3)*DM(4, 4)*DM(5, 5);
	TM(0, 2) = DM(2, 5)*DM(3, 5)*DM(4, 4) - DM(2, 5)*DM(3, 4)*DM(4, 5) - DM(2, 4)*DM(3, 5)*DM(5, 4) + \
		DM(2, 3)*DM(4, 5)*DM(5, 4) + DM(2, 4)*DM(3, 4)*DM(5, 5) - DM(2, 3)*DM(4, 4)*DM(5, 5);
	TM(1, 0) = DM(0, 5)*DM(3, 3)*DM(4, 5) - DM(0, 5)*DM(3, 5)*DM(4, 3) + DM(0, 4)*DM(3, 5)*DM(5, 3) - \
		DM(0, 3)*DM(4, 5)*DM(5, 3) - DM(0, 4)*DM(3, 3)*DM(5, 5) + DM(0, 3)*DM(4, 3)*DM(5, 5);
	TM(1, 1) = DM(1, 5)*DM(3, 3)*DM(4, 5) - DM(1, 5)*DM(3, 5)*DM(4, 3) + DM(1, 4)*DM(3, 5)*DM(5, 3) - \
		DM(1, 3)*DM(4, 5)*DM(5, 3) - DM(1, 4)*DM(3, 3)*DM(5, 5) + DM(1, 3)*DM(4, 3)*DM(5, 5);
	TM(1, 2) = DM(2, 5)*DM(3, 3)*DM(4, 5) - DM(2, 5)*DM(3, 5)*DM(4, 3) + DM(2, 4)*DM(3, 5)*DM(5, 3) - \
		DM(2, 3)*DM(4, 5)*DM(5, 3) - DM(2, 4)*DM(3, 3)*DM(5, 5) + DM(2, 3)*DM(4, 3)*DM(5, 5);
	TM(2, 0) = DM(0, 5)*DM(3, 4)*DM(4, 3) - DM(0, 5)*DM(3, 3)*DM(4, 4) - DM(0, 4)*DM(3, 4)*DM(5, 3) + \
		DM(0, 3)*DM(4, 4)*DM(5, 3) + DM(0, 4)*DM(3, 3)*DM(5, 4) - DM(0, 3)*DM(4, 3)*DM(5, 4);
	TM(2, 1) = DM(1, 5)*DM(3, 4)*DM(4, 3) - DM(1, 5)*DM(3, 3)*DM(4, 4) - DM(1, 4)*DM(3, 4)*DM(5, 3) + \
		DM(1, 3)*DM(4, 4)*DM(5, 3) + DM(1, 4)*DM(3, 3)*DM(5, 4) - DM(1, 3)*DM(4, 3)*DM(5, 4);
	TM(2, 2) = DM(2, 5)*DM(3, 4)*DM(4, 3) - DM(2, 5)*DM(3, 3)*DM(4, 4) - DM(2, 4)*DM(3, 4)*DM(5, 3) + \
		DM(2, 3)*DM(4, 4)*DM(5, 3) + DM(2, 4)*DM(3, 3)*DM(5, 4) - DM(2, 3)*DM(4, 3)*DM(5, 4);

	Ts = (-(DM(3, 5)*DM(4, 4)*DM(5, 3)) + DM(3, 4)*DM(4, 5)*DM(5, 3) + DM(3, 5)*DM(4, 3)*DM(5, 4) - \
		DM(3, 3)*DM(4, 5)*DM(5, 4) - DM(3, 4)*DM(4, 3)*DM(5, 5) + DM(3, 3)*DM(4, 4)*DM(5, 5));

	M(0, 0) = (DM(2, 0) + (DM(2, 3)*TM(0, 0) + DM(2, 4)*TM(1, 0) + DM(2, 5)*TM(2, 0)) / Ts) / 2.;
	M(0, 1) = (DM(2, 1) + (DM(2, 3)*TM(0, 1) + DM(2, 4)*TM(1, 1) + DM(2, 5)*TM(2, 1)) / Ts) / 2.;
	M(0, 2) = (DM(2, 2) + (DM(2, 3)*TM(0, 2) + DM(2, 4)*TM(1, 2) + DM(2, 5)*TM(2, 2)) / Ts) / 2.;
	M(1, 0) = -DM(1, 0) - (DM(1, 3)*TM(0, 0) + DM(1, 4)*TM(1, 0) + DM(1, 5)*TM(2, 0)) / Ts;
	M(1, 1) = -DM(1, 1) - (DM(1, 3)*TM(0, 1) + DM(1, 4)*TM(1, 1) + DM(1, 5)*TM(2, 1)) / Ts;
	M(1, 2) = -DM(1, 2) - (DM(1, 3)*TM(0, 2) + DM(1, 4)*TM(1, 2) + DM(1, 5)*TM(2, 2)) / Ts;
	M(2, 0) = (DM(0, 0) + (DM(0, 3)*TM(0, 0) + DM(0, 4)*TM(1, 0) + DM(0, 5)*TM(2, 0)) / Ts) / 2.;
	M(2, 1) = (DM(0, 1) + (DM(0, 3)*TM(0, 1) + DM(0, 4)*TM(1, 1) + DM(0, 5)*TM(2, 1)) / Ts) / 2.;
	M(2, 2) = (DM(0, 2) + (DM(0, 3)*TM(0, 2) + DM(0, 4)*TM(1, 2) + DM(0, 5)*TM(2, 2)) / Ts) / 2.;

	if (fabs(cv::determinant(M)) > 1.0e-10) {
		cv::Mat eVal, eVec;
		eigenNonSymmetric(M, eVal, eVec);

		// Select the eigen vector {a,b,c} which satisfies 4ac-b^2 > 0
		double cond[3];
		cond[0] = (4.0 * eVec.at<double>(0, 0) * eVec.at<double>(0, 2) - eVec.at<double>(0, 1) * eVec.at<double>(0, 1));
		cond[1] = (4.0 * eVec.at<double>(1, 0) * eVec.at<double>(1, 2) - eVec.at<double>(1, 1) * eVec.at<double>(1, 1));
		cond[2] = (4.0 * eVec.at<double>(2, 0) * eVec.at<double>(2, 2) - eVec.at<double>(2, 1) * eVec.at<double>(2, 1));
		if (cond[0]<cond[1]) {
			i = (cond[1]<cond[2]) ? 2 : 1;
		}
		else {
			i = (cond[0]<cond[2]) ? 2 : 0;
		}
		double norm = std::sqrt(eVec.at<double>(i, 0)*eVec.at<double>(i, 0) + eVec.at<double>(i, 1)*eVec.at<double>(i, 1) + eVec.at<double>(i, 2)*eVec.at<double>(i, 2));
		if (((eVec.at<double>(i, 0)<0.0 ? -1 : 1) * (eVec.at<double>(i, 1)<0.0 ? -1 : 1) * (eVec.at<double>(i, 2)<0.0 ? -1 : 1)) <= 0.0) {
			norm = -1.0*norm;
		}
		pVec(0) = eVec.at<double>(i, 0) / norm; pVec(1) = eVec.at<double>(i, 1) / norm; pVec(2) = eVec.at<double>(i, 2) / norm;

		//  Q = (TM . pVec)/Ts;
		Q(0, 0) = (TM(0, 0)*pVec(0) + TM(0, 1)*pVec(1) + TM(0, 2)*pVec(2)) / Ts;
		Q(0, 1) = (TM(1, 0)*pVec(0) + TM(1, 1)*pVec(1) + TM(1, 2)*pVec(2)) / Ts;
		Q(0, 2) = (TM(2, 0)*pVec(0) + TM(2, 1)*pVec(1) + TM(2, 2)*pVec(2)) / Ts;

		// assign raw solution
		rawSol.create(1, 6, CV_64F);
		rawSol.at<double>(0, 0) = pVec(0);
		rawSol.at<double>(0, 1) = pVec(1);
		rawSol.at<double>(0, 2) = pVec(2);
		rawSol.at<double>(0, 3) = Q(0, 0);
		rawSol.at<double>(0, 4) = Q(0, 1);
		rawSol.at<double>(0, 5) = Q(0, 2);

		// We compute the ellipse properties in the shifted coordinates as
		// doing so improves the numerical accuracy.
		/*
		double u1 = pVec(2)*Q(0, 0)*Q(0, 0)
		- pVec(1)*Q(0, 0)*Q(0, 1)
		+ pVec(0)*Q(0, 1)*Q(0, 1)
		+ pVec(1)*pVec(1)*Q(0, 2);

		double u2 = pVec(0)*pVec(2)*Q(0, 2);
		double l1 = sqrt(pVec(1)*pVec(1) + (pVec(0) - pVec(2))*(pVec(0) - pVec(2)));
		double l2 = pVec(0) + pVec(2);
		double l3 = pVec(1)*pVec(1) - 4 * pVec(0)*pVec(2);
		double p1 = 2 * pVec(2)*Q(0, 0) - pVec(1)*Q(0, 1);
		double p2 = 2 * pVec(0)*Q(0, 1) - pVec(1)*Q(0, 0);

		x0 = p1 / l3 + c.x;
		y0 = p2 / l3 + c.y;
		a = sqrt(2.)*sqrt((u1 - 4.0*u2) / ((l1 - l2)*l3));
		b = sqrt(2.)*sqrt(-1.0*((u1 - 4.0*u2) / ((l1 + l2)*l3)));
		if (pVec(1) == 0) {
		if (pVec(0)  < pVec(2)) {
		theta = 0;
		}
		else {
		theta = CV_PI / 2.;
		}
		}
		else {
		theta = CV_PI / 2. + 0.5*std::atan2(pVec(1), (pVec(0) - pVec(2)));
		}
		box.center.x = (float)x0;
		box.center.y = (float)y0;
		box.size.width = (float)(2.0*a);
		box.size.height = (float)(2.0*b);
		if (box.size.width > box.size.height)
		{
		float tmp;
		CV_SWAP(box.size.width, box.size.height, tmp);
		box.angle = (float)(fmod((90 + theta * 180 / CV_PI), 180.0));
		}
		else {
		box.angle = (float)(fmod(theta * 180 / CV_PI, 180.0));
		};
		if (box.size.width > box.size.height)
		{
		float tmp;
		CV_SWAP(box.size.width, box.size.height, tmp);
		box.angle = (float)(fmod((90 + theta * 180 / CV_PI), 180.0));
		}
		else {
		box.angle = (float)(fmod(theta * 180 / CV_PI, 180.0));
		};
		*/

		double A = pVec(0);
		double B = pVec(1);
		double C = pVec(2);
		double D = Q(0, 0);
		double E = Q(0, 1);
		double F = Q(0, 2);

		double xc = (2 * C*D - B*E) / (B*B - 4 * A*C);
		double yc = (2 * A*E - B*D) / (B*B - 4 * A*C);
		double a = -sqrt(2 * (A*E*E + C*D*D - B*D*E + (B*B - 4 * A*C)*F)*(A + C + (sqrt((A - C)*(A - C) + B*B)))) / (B*B - 4 * A*C);
		double b = -sqrt(2 * (A*E*E + C*D*D - B*D*E + (B*B - 4 * A*C)*F)*(A + C - (sqrt((A - C)*(A - C) + B*B)))) / (B*B - 4 * A*C);

		box.center.x = (float)xc;
		box.center.y = (float)yc;
		box.size.width = (float)(2.0*a);
		box.size.height = (float)(2.0*b);
		double theta = 0;
		if (B == 0 && A<C)
			theta = 0;
		else if (B == 0 && A>C)
			theta = CV_PI / 2;
		else
			theta = atan2(C - A - sqrt((A - C)*(A - C) + B*B), B);

		box.angle = theta;
	}
	else {
		box = cv::fitEllipse(points);
	}
	return box;
}

void fromRotatedRectToEllipseParams(const cv::RotatedRect &rect, cv::Mat &et)
{
	et.create(1, 6, CV_64F);

	float major = rect.size.width*0.5;
	float minor = rect.size.height*0.5;

	float xc = rect.center.x;
	float yc = rect.center.y;

	float theta = rect.angle * CV_PI / 180.0;

	float cy = cos(theta);
	float sy = sin(theta);
	float cy2 = cy*cy;
	float sy2 = sy*sy;
	float sycy = sy*cy;
	float b2 = minor*minor;
	float a2 = major*major;

	float a2b2_inv = 1;

	double A, B, C, D, E, F;
	A = (cy2*b2 + sy2*a2)*a2b2_inv;
	B = 2 * (+sycy*b2 - sycy*a2)*a2b2_inv;
	C = (sy2*b2 + cy2*a2)*a2b2_inv;
	D = -2 * A*xc - B*yc;
	E = -B*xc - 2 * C*yc;
	F = A*xc*xc + B*xc*yc + C*yc*yc - a2*b2;

	et.at<double>(0, 0) = A;
	et.at<double>(0, 1) = B;
	et.at<double>(0, 2) = C;
	et.at<double>(0, 3) = D;
	et.at<double>(0, 4) = E;
	et.at<double>(0, 5) = F;
}

cv::Vec3f estimatePositionAnalyticalSol(const cv::Mat &et, const cv::Mat &camMatrix, float diameter)
{
	// pose estimation using eigen decomposition, analytical solution
	cv::Mat Q(3, 3, CV_64F);
	float inv_f = 1. / camMatrix.at<double>(0, 0);
	//std::cout << et << std::endl;
	Q.at<double>(0, 0) = et.at<double>(0, 0);
	Q.at<double>(0, 1) = et.at<double>(0, 1)*0.5;
	Q.at<double>(0, 2) = et.at<double>(0, 3)*inv_f*0.5;

	Q.at<double>(1, 0) = et.at<double>(0, 1)*0.5;
	Q.at<double>(1, 1) = et.at<double>(0, 2);
	Q.at<double>(1, 2) = et.at<double>(0, 4)*inv_f*0.5;

	Q.at<double>(2, 0) = Q.at<double>(0, 2);
	Q.at<double>(2, 1) = Q.at<double>(1, 2);
	Q.at<double>(2, 2) = et.at<double>(0, 5)*inv_f*inv_f;


	//std::cout << Q << std::endl;

	cv::Vec3d eval;
	cv::Mat evec(3, 3, CV_64F);
	cv::eigen(Q, eval, evec);
	//std::cout << eval << std::endl;
	int l1 = 1, l2 = 0, l3 = 2;
	float e2e3 = eval[l2] * eval[l3];
	float e2_e3_inv = 1. / (eval[l2] - eval[l3]);
	float e2_e1 = eval[l2] - eval[l1];
	float e1_e3 = eval[l1] - eval[l3];

	float d1 = e2_e1 * e2_e3_inv;
	float d2 = e1_e3 * e2_e3_inv;
	
	const float radius = diameter * 0.5;

	// theoratically, one marker, there will be an ambiguity which cannot be solved, 
	// but normally, these two results are quite similar, that's why whycon system only select one
	// TODO, following ECCV, use multiple markers to refine the sign and select the most apprepriate solution
	cv::Vec3f t;
	t(0) = 0;
	t(1) = 0;
	t(2) = 0;

	if (d1 > 0 && d2 > 0 && e2e3 < 0)
	{
		cv::Mat twc1(1, 3, CV_64F);
		float scale = 1 / sqrt(-e2e3) * radius;
		twc1 = scale* eval[l3] * sqrt(d1)*evec.row(l2)
			+ scale* eval[l2] * sqrt(d2)*evec.row(l3);
		std::cout << "++" << twc1 << std::endl;
		//twc = -1 * scale* eval[l3] * sqrt(d1)*evec.row(l2)
		//	+ scale* eval[l2] * sqrt(d2)*evec.row(l3);
		//std::cout << "-+" << twc << std::endl;

		cv::Mat twc2(1, 3, CV_64F);

		//twc = 1 * scale* eval[l3] * sqrt(d1)*evec.row(l2)
		//	- scale* eval[l2] * sqrt(d2)*evec.row(l3);
		//std::cout << "+-" << twc << std::endl;
		twc2 = -1 * scale* eval[l3] * sqrt(d1)*evec.row(l2)
			- scale* eval[l2] * sqrt(d2)*evec.row(l3);
		std::cout << "--" << twc2 << std::endl;

		if (twc1.ptr<double>(0)[2] > 0)
		{
			t(0) = (float)(twc1.ptr<double>(0)[0]);
			t(1) = (float)(twc1.ptr<double>(0)[1]);
			t(2) = (float)(twc1.ptr<double>(0)[2]);
		}
		else
		{
			t(0) = (float)(twc2.ptr<double>(0)[0]);
			t(1) = (float)(twc2.ptr<double>(0)[1]);
			t(2) = (float)(twc2.ptr<double>(0)[2]);
		}
	}

	return t;
}

cv::Vec3f estimatePositionGeometricSol(const cv::RotatedRect &rect, const cv::Mat &camMatrix, float diameter)
{
	// try geometric solution based on TRO 1999

	// step1, compute 4 points in image frame
	float A = rect.size.width * 0.5;
	float B = rect.size.height * 0.5;

	if (A < B)std::swap(A, B);

	float xc = rect.center.x;
	float yc = rect.center.y;
	float theta = rect.angle * CV_PI / 180.0;
	float sy = sin(theta); float cy = cos(theta);
	cv::Point2f p, q, b, c;
	p.x = xc + cy * A;
	p.y = yc + sy * A;
	q.x = xc - cy * A;
	q.y = yc - sy * A;
	b.x = xc + sy * B;
	b.y = yc - cy * B;
	c.x = xc - sy * B;
	c.y = yc + cy * B;
	float f = camMatrix.at<double>(0, 0);
	cv::Vec3f v1 = cv::Vec3f(b.x, b.y, f);
	cv::Vec3f v2 = cv::Vec3f(c.x, c.y, f);
	cv::Vec3f v3 = cv::Vec3f(p.x, p.y, f);
	cv::Vec3f v4 = cv::Vec3f(q.x, q.y, f);
	v1 = cv::normalize(v1);
	v2 = cv::normalize(v2);
	v3 = cv::normalize(v3);
	v4 = cv::normalize(v4);

	//std::cout << cv::norm(v1, CV_L2) << std::endl;
	//std::cout << v1 << std::endl;
	//std::cout << v2 << std::endl;
	//std::cout << v3 << std::endl;
	//std::cout << v4 << std::endl;
	//std::cout << v1.dot(v2) << std::endl;

	float angbc = acos(v1.dot(v2));
	float angpq = acos(v3.dot(v4));

	float vb1 = f * tan(angbc*0.5);
	float va1 = f * tan(angpq*0.5);

	float tmp1 = vb1*vb1;
	float tmp2 = va1*va1;
	float f2 = f*f;

	const float R = diameter * 0.5;
	float alpha1 = (sqrt((tmp2 + f2) / (tmp1 + f2)) + sqrt((tmp2 - tmp1) / (tmp1 + f2))) / va1;
	float beta1 = (sqrt((tmp2 + f2) / (tmp1 + f2)) - sqrt((tmp2 - tmp1) / (tmp1 + f2))) / va1;
	float gamma1 = sqrt((tmp2 + f2) / (tmp1 + f2)) / va1;

	float alpha2 = beta1;
	float beta2 = alpha1;
	float gamma2 = gamma1;

	float depth = R * (alpha1 + beta1) * 0.5 * f;
	//std::cout << "x : " << depth * xc / f << "; y : " << depth * yc / f << "; z : " << depth << std::endl;
	//std::cout << "B:" <<  (alpha1) * R * vb1 << ";" << alpha1 * f * R << std::endl;
	//std::cout << "C:" << -(beta1)* R * vb1 << ";" << beta1 * f * R  << std::endl;
	//std::cout << "P:" << gamma1 * va1 * R << ";" << gamma1 * f * R << std::endl;

	cv::Vec3f t;
	t(0) = depth * xc / f;
	t(1) = depth * yc / f;
	t(2) = depth;

	return t;
}




