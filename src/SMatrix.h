///////////////////////////////////////////////////////////////////////////////////////
// SMatrix.h
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

#ifndef SMATRIX_H
#define SMATRIX_H


#include "DMatrix.h"
#include "DVector.h"


#define SM_CMAX 1000


class CDMatrix;
class CDVector;


class CSMatrix
{
public:
	CSMatrix();
	CSMatrix(int rows, int cols);
	CSMatrix(int rows, int cols, int nz_max);
	virtual ~CSMatrix();

	void Init(int rows, int cols);
	void Init(int rows, int cols, int nz_max);
	void Free();

	void Add(int r, int c, double val);
	void Set(double alpha, CDMatrix& X);

	void GenerateCompInfo();

	void Set_APB(double alpha, CSMatrix& A, CSMatrix& B);
	void Set_ATA(double alpha, CSMatrix& A);

	void Mult(double s);

	void GenerateLU();
	void SolveLU(CDVector& b, CDVector& x);

	BOOL SolveLinearSystem(CDVector& b, CDVector& x);
	BOOL SolveLinearSystem2(CDVector& b1, CDVector& x1, CDVector& b2, CDVector& x2);
	BOOL SolveLinearSystem3(CDVector& b1, CDVector& x1, CDVector& b2, CDVector& x2, CDVector& b3, CDVector& x3);

	void SaveMatrix(const char* filename, int mode = 0);
	void SaveLUMatrix(const char* filename_L, const char* filename_U, int mode = 0);

public:
	int m_rows;
	int m_cols;
	//
	int m_nz;
	int m_nz_max;
	//
	double* m_val;
	int* m_rowind;
	int* m_colind;
	//
	double* m_cc_val;
	int* m_cc_rowind;
	int* m_cc_colptr;
	//
	double* m_cr_val;
	int* m_cr_rowptr;
	int* m_cr_colind;
	//
	int* m_indx;
	CDMatrix* m_LU;
};


#endif
