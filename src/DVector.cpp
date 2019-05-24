///////////////////////////////////////////////////////////////////////////////////////
// DVector.cpp
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
#include "DVector.h"
#include "MyUtils.h"


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CDVector::CDVector()
{
	m_dv = NULL;
	m_rows = 0;
}

CDVector::CDVector(int rows)
{
	m_dv = NULL;
	Init(rows);
}

CDVector::~CDVector()
{
	Free();
}

void CDVector::Init(int rows)
{
	Free();

	m_rows = rows;

	m_dv = (double*)MyAllocEx(m_rows*sizeof(double), "CDVector::Init m_dv");
	Clear();
}
	
void CDVector::Free()
{
	if (m_dv) {
		MyFree(m_dv);
		m_dv = NULL;
	}

	m_rows = 0;
}

void CDVector::Clear()
{
	int i;
	double* p = m_dv;
	for (i = 0; i < m_rows; i++) {
		(*p++) = 0;
	}
}

void CDVector::Set(CDVector& X)
{
	if (m_rows != X.m_rows) {
		Init(X.m_rows);
	}

	int i;
	double* p = X.m_dv;
	double* q = m_dv;
	for (i = 0; i < m_rows; i++) {
		*q++ = *p++;
	}
}
void CDVector::SetMinus(CDVector& X)
{
	int i;
	double* p = X.m_dv;
	double* q = m_dv;
	for (i = 0; i < m_rows; i++) {
		*q++ = -(*p++);
	}
}

void CDVector::Set_APb(double alpha, CSMatrix& A, CDVector& b)
{
	Clear();
	int j;
	if (alpha == 1.0) {
		for (j = 0; j < A.m_nz; j++) {
			m_dv[A.m_rowind[j]] += A.m_val[j] * b.m_dv[A.m_colind[j]];
		}
	} else {
		for (j = 0; j < A.m_nz; j++) {
			m_dv[A.m_rowind[j]] += alpha * A.m_val[j] * b.m_dv[A.m_colind[j]];
		}
	}
}

void CDVector::Set_APb(double alpha, CDMatrix& A, CDVector& b)
{
	Clear();
	int i, j;
	if (alpha == 1.0) {
		for (j = 0; j < A.m_rows; j++) {
			for (i = 0; i < A.m_cols; i++) {
				m_dv[j] += A.m_dm[j][i] * b.m_dv[i];
			}
		}
	} else {
		for (j = 0; j < A.m_rows; j++) {
			for (i = 0; i < A.m_cols; i++) {
				m_dv[j] += alpha * A.m_dm[j][i] * b.m_dv[i];
			}
		}
	}
}

void CDVector::Add_APb(double alpha, CSMatrix& A, CDVector& b)
{
/*
	CDVector y(m_rows);
	int j;
	if (alpha == 1.0) {
		for (j = 0; j < A.m_nz; j++) {
			y.m_dv[A.m_rowind[j]] += A.m_val[j] * b.m_dv[A.m_colind[j]];
		}
	} else {
		for (j = 0; j < A.m_nz; j++) {
			y.m_dv[A.m_rowind[j]] += alpha * A.m_val[j] * b.m_dv[A.m_colind[j]];
		}
	}
	Add(y);
/*/
	int j;
	if (alpha == 1.0) {
		for (j = 0; j < A.m_nz; j++) {
			m_dv[A.m_rowind[j]] += A.m_val[j] * b.m_dv[A.m_colind[j]];
		}
	} else {
		for (j = 0; j < A.m_nz; j++) {
			m_dv[A.m_rowind[j]] += alpha * A.m_val[j] * b.m_dv[A.m_colind[j]];
		}
	}
//*/
}

void CDVector::Add(CDVector& X)
{
	int i;
	double* p = X.m_dv;
	double* q = m_dv;
	for (i = 0; i < m_rows; i++) {
		(*q) = (*q) + (*p);
		p++;
		q++;
	}
}

void CDVector::Mult(double s)
{
	int i;
	double* p = m_dv;
	for (i = 0; i < m_rows; i++) {
		(*p) = s * (*p);
		p++;
	}
}

void CDVector::Set_a_b(double alpha, CDVector& a, double beta, CDVector& b)
{
	int i;
	double* p1 = a.m_dv;
	double* p2 = b.m_dv;
	double* q  = m_dv;
	if (alpha == 1.0) {
		if (beta == 1.0) {
			for (i = 0; i < m_rows; i++) {
				(*q++) = (*p1++) + (*p2++);
			}
		} else {
			for (i = 0; i < m_rows; i++) {
				(*q++) = (*p1++) + beta * (*p2++);
			}
		}
	} else {
		if (beta == 1.0) {
			for (i = 0; i < m_rows; i++) {
				(*q++) = alpha * (*p1++) + (*p2++);
			}
		} else {
			for (i = 0; i < m_rows; i++) {
				(*q++) = alpha * (*p1++) + beta * (*p2++);
			}
		}
	}
}

double CDVector::Get_ata()
{
	int i;
	double val = 0.0;
	double* p = m_dv;
	for (i = 0; i < m_rows; i++) {
		val += (*p) * (*p);
		p++;
	}
	return val;
}

double CDVector::Get_atb(CDVector& b)
{
	int i;
	double val = 0.0;
	double* p1 = m_dv;
	double* p2 = b.m_dv;
	for (i = 0; i < m_rows; i++) {
		val += (*p1++) * (*p2++);
	}
	return val;
}

double CDVector::Get_atAPb(double alpha, CSMatrix& A, CDVector& b)
{
	CDVector y(A.m_rows);
	y.Set_APb(alpha, A, b);
	return Get_atb(y);
}

BOOL CDVector::SaveVector(const char* filename, int mode)
{
	int i;
	if (mode == 0) {
		FILE *fp;
		fp = fopen(filename, "wb");
		if (fp == NULL) {
			return FALSE;
		}
		fwrite(m_dv, sizeof(double), m_rows, fp);
		fclose(fp);
	} else if (mode == 1) {
		FILE *fp;
		fp = fopen(filename, "w");
		if (fp == NULL) {
			return FALSE;
		}
		for (i = 0; i < m_rows; i++) {
			fprintf(fp, "%e\r\n", m_dv[i]);
		}
		fclose(fp);
	} else {
		FILE *fp;
		fp = fopen(filename, "w");
		if (fp == NULL) {
			return FALSE;
		}
		for (i = 0; i < m_rows; i++) {
			if (m_dv[i] == 0) {
				fprintf(fp, "0\r\n");
			} else {
				fprintf(fp, "1\r\n");
			}
		}
		fclose(fp);
	}

	return TRUE;
}

BOOL CDVector::LoadVector(const char* filename)
{
	FILE *fp;
	fp = fopen(filename, "rb");
	if (fp == NULL) {
		return FALSE;
	}
	fread(m_dv, sizeof(double), m_rows, fp);
	fclose(fp);
	return TRUE;
}
