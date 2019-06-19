///////////////////////////////////////////////////////////////////////////////////////
// DMatrix.cpp
// Developed by Dongjin Kwon
///////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2011-2014 Dongjin Kwon
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "DMatrix.h"
#include "MyUtils.h"
//
#include <math.h>


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CDMatrix::CDMatrix()
{
	m_dm = NULL;
}

CDMatrix::CDMatrix(int rows, int cols)
{
	m_dm = NULL;
	Init(rows, cols);
}

CDMatrix::CDMatrix(int rows, int cols, CSMatrix* X, BOOL transpose)
{
	m_dm = NULL;
	Init(rows, cols, X, transpose);
}

CDMatrix::~CDMatrix()
{
	Free();
}

void CDMatrix::Init(int rows, int cols, CSMatrix* X, BOOL transpose)
{
	Free();

	int i, j;

	if (X == NULL) {
		m_rows = rows;
		m_cols = cols;
	} else {
		if (!transpose) {
			m_rows = rows;
			m_cols = cols;
		} else {
			m_rows = cols;
			m_cols = rows;
		}
	}

	m_dm = (double**)MyAllocEx(m_rows*sizeof(double*), "CDMatrix::Init m_dm");
	for (j = 0; j < m_rows; j++) {
		m_dm[j] = (double*)MyAllocEx(m_cols*sizeof(double), "CDMatrix::Init m_dm[j]");
		memset(m_dm[j], 0, m_cols*sizeof(double));
		//for (i = 0; i < m_cols; i++) {
		//	m_dm[j][i] = 0.0;
		//}
	}

	if (X != NULL) {
		if (!transpose) {
			for (i = 0; i < X->m_nz; i++) {
				m_dm[X->m_rowind[i]][X->m_colind[i]] = X->m_val[i];
			}
		} else {
			for (i = 0; i < X->m_nz; i++) {
				m_dm[X->m_colind[i]][X->m_rowind[i]] = X->m_val[i];
			}
		}
	}
}
	
void CDMatrix::Free()
{
	int j;

	if (m_dm) {
		for (j = 0; j < m_rows; j++) {
			MyFree(m_dm[j]);
		}
		MyFree(m_dm);
		m_dm = NULL;
	}

	m_cols = m_rows = 0;
}

void CDMatrix::Clear()
{
	int i, j;
	for (j = 0; j < m_rows; j++) {
		double* p_dm = m_dm[j];
		for (i = 0; i < m_cols; i++) {
			*p_dm++ = 0;
		}
	}
}

void CDMatrix::Set(double alpha, CSMatrix& X)
{
	int j;
	
	//Clear();

	if (alpha == 1.0) {
		for (j = 0; j < X.m_nz; j++) {
			m_dm[X.m_rowind[j]][X.m_colind[j]] = X.m_val[j];
		}
	} else {
		for (j = 0; j < X.m_nz; j++) {
			m_dm[X.m_rowind[j]][X.m_colind[j]] = alpha * X.m_val[j];
		}
	}
}

void CDMatrix::Set(double alpha, CDMatrix& X)
{
	int i, j;
	
	//Clear();

	if (alpha == 1.0) {
		for (j = 0; j < m_rows; j++) {
			double* p_dm = X.m_dm[j];
			double* q_dm = m_dm[j];
			for (i = 0; i < m_cols; i++) {
				*q_dm++ = *p_dm++;
			}
		}
	} else {
		for (j = 0; j < m_rows; j++) {
			double* p_dm = X.m_dm[j];
			double* q_dm = m_dm[j];
			for (i = 0; i < m_cols; i++) {
				*q_dm++ = alpha * (*p_dm++);
			}
		}
	}
}

void CDMatrix::Add_APAT(double alpha, CSMatrix& A)
{
/*
	int i, j;
	CDMatrix AT(A.m_rows, A.m_cols, &A, TRUE);
	if (alpha == 1.0) {
		for (i = 0; i < A.m_rows; i++) {
			for (j = 0; j < A.m_nz; j++) {
				m_dm[A.m_rowind[j]][i] += A.m_val[j] * AT.m_dm[A.m_colind[j]][i];
			}
		}
	} else {
		for (i = 0; i < A.m_rows; i++) {
			for (j = 0; j < A.m_nz; j++) {
				m_dm[A.m_rowind[j]][i] += alpha * A.m_val[j] * AT.m_dm[A.m_colind[j]][i];
			}
		}
	}
/*/
	int i, j, k, cp1, cp2;
	if (alpha == 1.0) {
		for (k = 0; k < A.m_cols; k++) {
			cp1 = A.m_cc_colptr[k];
			cp2 = A.m_cc_colptr[k+1];
			for (j = cp1; j < cp2; j++) {
				for (i = cp1; i < cp2; i++) {
					m_dm[A.m_cc_rowind[j]][A.m_cc_rowind[i]] += A.m_cc_val[j] * A.m_cc_val[i];
				}
			}
		}
	} else {
		for (k = 0; k < A.m_cols; k++) {
			cp1 = A.m_cc_colptr[k];
			cp2 = A.m_cc_colptr[k+1];
			for (j = cp1; j < cp2; j++) {
				for (i = cp1; i < cp2; i++) {
					m_dm[A.m_cc_rowind[j]][A.m_cc_rowind[i]] += alpha * A.m_cc_val[j] * A.m_cc_val[i];
				}
			}
		}
	}
//*/
}

void CDMatrix::Set_APB(double alpha, CSMatrix& A, CSMatrix& B)
{
/*
	int i, j;
	CDMatrix dm_B(B.m_rows, B.m_cols, &B, FALSE);
	if (alpha == 1.0) {
		for (i = 0; i < A.m_rows; i++) {
			for (j = 0; j < A.m_nz; j++) {
				m_dm[A.m_rowind[j]][i] += A.m_val[j] * dm_B.m_dm[A.m_colind[j]][i];
			}
		}
	} else {
		for (i = 0; i < A.m_rows; i++) {
			for (j = 0; j < A.m_nz; j++) {
				m_dm[A.m_rowind[j]][i] += alpha * A.m_val[j] * dm_B.m_dm[A.m_colind[j]][i];
			}
		}
	}
/*/
	int i, j, k;
	int i_cp1, i_cp2, j_cp1, j_cp2;
	if (alpha == 1.0) {
		for (k = 0; k < A.m_cols; k++) {
			j_cp1 = A.m_cc_colptr[k];
			j_cp2 = A.m_cc_colptr[k+1];
			i_cp1 = B.m_cr_rowptr[k];
			i_cp2 = B.m_cr_rowptr[k+1];
			for (j = j_cp1; j < j_cp2; j++) {
				for (i = i_cp1; i < i_cp2; i++) {
					m_dm[A.m_cc_rowind[j]][B.m_cr_colind[i]] += A.m_cc_val[j] * B.m_cr_val[i];
				}
			}
		}
	} else {
		for (k = 0; k < A.m_cols; k++) {
			j_cp1 = A.m_cc_colptr[k];
			j_cp2 = A.m_cc_colptr[k+1];
			i_cp1 = B.m_cr_rowptr[k];
			i_cp2 = B.m_cr_rowptr[k+1];
			for (j = j_cp1; j < j_cp2; j++) {
				for (i = i_cp1; i < i_cp2; i++) {
					m_dm[A.m_cc_rowind[j]][B.m_cr_colind[i]] += alpha * A.m_cc_val[j] * B.m_cr_val[i];
				}
			}
		}
	}
//*/
}

void CDMatrix::Add(double alpha, CDMatrix& X)
{
	int i, j;
	for (j = 0; j < m_rows; j++) {
		double* p = X.m_dm[j];
		double* q = m_dm[j];
		for (i = 0; i < m_cols; i++) {
			(*q) = (*q) + alpha * (*p);
			p++;
			q++;
		}
	}
}

void CDMatrix::Mult(double s)
{
	int i, j;
	for (j = 0; j < m_rows; j++) {
		double* p = m_dm[j];
		for (i = 0; i < m_cols; i++) {
			(*p) = s * (*p);
			p++;
		}
	}
}

void CDMatrix::RowOpSwap(int r1, int r2)
{
    for (int i = 0; i < m_cols; i++) {
        double temp = m_dm[r1][i];
        m_dm[r1][i] = m_dm[r2][i];
        m_dm[r2][i] = temp;
    }
}
void CDMatrix::RowOpPlus(int r1, int r2)
{
    for (int i = 0; i < m_cols; i++) {
        m_dm[r1][i] += m_dm[r2][i];
    }
}
void CDMatrix::RowOpMinus(int r1, int r2)
{
    for (int i = 0; i < m_cols; i++) {
        m_dm[r1][i] -= m_dm[r2][i];
    }
}
void CDMatrix::RowOpMultipliedPlus(int r1, int r2, double a)
{
    for (int i = 0; i < m_cols; i++) {
        m_dm[r1][i] += a * m_dm[r2][i];
    }
}
void CDMatrix::RowOpMultipliedMinus(int r1, int r2, double a)
{
    for (int i = 0; i < m_cols; i++) {
        m_dm[r1][i] -= a * m_dm[r2][i];
    }
}
void CDMatrix::RowOpMultiply(int r, double a)
{
    for (int i = 0; i < m_cols; i++) {
        m_dm[r][i] *= a;
    }
}
void CDMatrix::RowOpDivide(int r, double a)
{
    RowOpMultiply(r, 1.0 / a);
}
BOOL CDMatrix::MakeInverse(CDMatrix& X)
{
 	int i, j;

    if (m_rows != m_cols) {
       TRACE("---------- ERROR! Use pseudo-inverse function\n");
       return FALSE;
    }

	X.Clear();
	for (i = 0; i < m_rows; i++) {
		X.m_dm[i][i] = 1.0;
	}

    // Make lower tirangle elements to zero
    for (i = 0; i < m_cols; i++) {
        // Make i-th diagonal element to 1
        if (m_dm[i][i] == 0.0) {
            for (j = i + 1; j < m_rows; j++) {
                if (m_dm[j][i] != 0.0) {
                    X.RowOpSwap(i, j);
                    RowOpSwap(i, j);
                    break;
                }
            }
        }

        if (m_dm[i][i] != 0.0) {
            X.RowOpDivide(i, m_dm[i][i]);
            RowOpDivide(i, m_dm[i][i]);
        } else {
            TRACE("---------- ERROR! Rank is not fit (Check %d-th column)\n", i);
        }

        // Make i-th column to zero only in lower triangle elements
        for (int j = i + 1; j < m_rows; j++) {
            X.RowOpMultipliedMinus(j, i, m_dm[j][i]);
            RowOpMultipliedMinus(j, i, m_dm[j][i]);
        }
    }
    // Make upper triangle elements to zero
    for (i = m_rows - 1; i > 0; i--) {
        for (j = 0; j < i; j++) {
            X.RowOpMultipliedMinus(j, i, m_dm[j][i]);
            RowOpMultipliedMinus(j, i, m_dm[j][i]);
        }
    }

    return TRUE;
}

//////////////////////////////////////////////////////////////////////
#define TINY 1e-20
void ludcmp(double **a, int n, int *indx, double *d)
{
	int i, imax, j, k;
	double big, dum, sum, temp;
	double *vv;

	vv = (double*)MyAlloc(n*sizeof(double));
	*d = 1.0;
	for (i = 0; i < n; i++) {
		big = 0.0;
		for (j = 0; j < n; j++) {
			if ((temp = fabs(a[i][j])) > big) big = temp;
		}
		if (big == 0.0) {
			TRACE("Singular matrix in routine ludcmp\r\n");
			return;
		}
		vv[i] = 1.0 / big;
	}
	for (j = 0; j < n; j++) {
		for (i = 0; i < j; i++) {
			sum = a[i][j];
			for (k = 0; k < i; k++) sum -= a[i][k]*a[k][j];
			a[i][j] = sum;
		}
		big = 0.0;
		imax = 0;
		for (i = j; i < n; i++) {
			sum = a[i][j];
			for (k = 0; k < j; k++) sum -= a[i][k]*a[k][j];
			a[i][j] = sum;
			if ((dum = vv[i]*fabs(sum)) >= big) {
				big = dum;
				imax = i;
			}
		}
		if (j != imax) {
			for (k = 0; k < n; k++) {
				dum = a[imax][k];
				a[imax][k] = a[j][k];
				a[j][k] = dum;
			}
			*d = -(*d);
			vv[imax] = vv[j];
		}
		indx[j] = imax;
		if (a[j][j] == 0.0) a[j][j] = TINY;
		if (j != n-1) {
			dum = 1.0 / (a[j][j]);
			for (i = j+1; i < n; i++) a[i][j] *= dum;
		}
	}
	MyFree(vv);
}
void lubksb(double **a, int n, int *indx, double b[])
{
	int i, ii = 0, ip, j;
	double sum;

	for (i = 0; i < n; i++) {
		ip = indx[i];
		sum = b[ip];
		b[ip] = b[i];
		if (ii != 0) {
			for (j = ii-1; j < i; j++) sum -= a[i][j]*b[j];
		} else if (sum != 0.0) {
			ii = i+1;
		}
		b[i] = sum;
	}
	for (i = n-1; i >= 0; i--) {
		sum = b[i];
		for (j = i+1; j < n; j++) sum -= a[i][j]*b[j];
		b[i] = sum / a[i][i];
	}
}

void CDMatrix::GetLUDMatrix(CDMatrix &LU, int* indx) {
	double d;
	int i, j;
	for (j = 0; j < m_rows; j++) {
		double* p_dm = m_dm[j];
		double* q_dm = LU.m_dm[j];
		for (i = 0; i < m_cols; i++) {
			*q_dm++ = *p_dm++;
		}
	}
	ludcmp(LU.m_dm, m_rows, indx, &d);
}

void CDMatrix::GetLUDSolution(CDMatrix &LU, int* indx, CDVector &b, CDVector &x) {
	int i;
	for (i = 0; i < b.m_rows; i++) {
		x.m_dv[i] = b.m_dv[i];
	}
	lubksb(LU.m_dm, LU.m_rows, indx, x.m_dv);
}
//////////////////////////////////////////////////////////////////////

BOOL CDMatrix::SaveMatrix(const char* filename, int mode)
{
	int i, j;
	FILE *fp;
	if (mode == 0) {
		fp = fopen(filename, "wb");
		if (fp == NULL) {
			return FALSE;
		}
		for (j = 0; j < m_rows; j++) {
			fwrite(m_dm[j], sizeof(double), m_cols, fp);
		}
	} else if (mode == 1) {
		fp = fopen(filename, "w");
		if (fp == NULL) {
			return FALSE;
		}
		for (j = 0; j < m_rows; j++) {
			for (i = 0; i < m_cols; i++) {
				fprintf(fp, "%e ", m_dm[j][i]);
			}
			fprintf(fp, "\n");
		}
	} else {
		fp = fopen(filename, "w");
		if (fp == NULL) {
			return FALSE;
		}
		for (j = 0; j < m_rows; j++) {
			for (i = 0; i < m_cols; i++) {
				if (m_dm[j][i] == 0) {
					fprintf(fp, "0 ");
				} else {
					fprintf(fp, "1 ");
				}
			}
			fprintf(fp, "\n");
		}
	}
	fclose(fp);

	return TRUE;
}

BOOL CDMatrix::LoadMatrix(const char* filename, int mode)
{
	int i, j;
	FILE *fp = NULL;
	if (mode == 0) {
		fp = fopen(filename, "rb");
		if (fp == NULL) {
			return FALSE;
		}
		for (j = 0; j < m_rows; j++) {
			fread(m_dm[j], sizeof(double), m_cols, fp);
		}
	} else if (mode == 1) {
		fp = fopen(filename, "r");
		if (fp == NULL) {
			return FALSE;
		}
		for (j = 0; j < m_rows; j++) {
			for (i = 0; i < m_cols; i++) {
				fscanf(fp, "%lf ", &m_dm[j][i]);
			}
			fscanf(fp, "\n");
		}
	}
	if (fp != NULL) {
		fclose(fp);
	}

	return TRUE;
}
