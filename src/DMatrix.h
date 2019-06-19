///////////////////////////////////////////////////////////////////////////////////////
// DMatrix.h
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

#ifndef DMATRIX_H
#define DMATRIX_H


#include "SMatrix.h"
#include "DVector.h"


class CSMatrix;
class CDVector;


class CDMatrix  
{
public:
	CDMatrix();
	CDMatrix(int rows, int cols);
	CDMatrix(int rows, int cols, CSMatrix* X, BOOL transpose);
	virtual ~CDMatrix();

	void Init(int rows, int cols, CSMatrix* X = NULL, BOOL transpose = FALSE);
	void Free();
	void Clear();

	void Set(double alpha, CSMatrix& X);
	void Set(double alpha, CDMatrix& X);

	void Add_APAT(double alpha, CSMatrix& A);
	void Set_APB(double alpha, CSMatrix& A, CSMatrix& B);

	void Add(double alpha, CDMatrix& X);
	void Mult(double s);

	void RowOpSwap(int r1, int r2);
	void RowOpPlus(int r1, int r2);
	void RowOpMinus(int r1, int r2);
	void RowOpMultipliedPlus(int r1, int r2, double a);
	void RowOpMultipliedMinus(int r1, int r2, double a);
	void RowOpMultiply(int r, double a);
	void RowOpDivide(int r, double a);
	BOOL MakeInverse(CDMatrix& X);

	void GetLUDMatrix(CDMatrix &LU, int* indx);
	void GetLUDSolution(CDMatrix &LU, int* indx, CDVector &b, CDVector &x);

	BOOL SaveMatrix(const char* filename, int mode = 0);
	BOOL LoadMatrix(const char* filename, int mode = 0);

public:
	double** m_dm;
	int m_rows;
	int m_cols;
};


#endif
