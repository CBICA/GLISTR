///////////////////////////////////////////////////////////////////////////////////////
// VolRegOpt.h
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

#include "ENT_table.h"

#define NMI_COLOR_NUM 32
#define NMI_COLOR_SHIFT 3
#define NMI_COLOR_MUL 0.125
#define DC_REAL float


void get_ninv(DC_REAL*** ninv, int mesh_ex, int mesh_ey, int mesh_ez);

template <class T>
void ComputeDataCost3D_NCC(T**** vdata1, int vd1_x, int vd1_y, int vd1_z, int vd1_s, T**** vdata2, int vd2_x, int vd2_y, int vd2_z, int vd2_s,
	REALV**** X1, REALV**** Y1, REALV**** Z1, REALV**** dX1, REALV**** dY1, REALV**** dZ1, REALV**** X2, REALV**** Y2, REALV**** Z2, REALV**** dX2, REALV**** dY2, REALV**** dZ2,
	int mesh_x, int mesh_y, int mesh_z, int mesh_ex, int mesh_ey, int mesh_ez,
	REALV* disp_x, REALV* disp_y, REALV* disp_z, int num_d, REALV**** dcv,
	int dc_weight, int dc_skip_back, T dc_back_color, int ninv_s = 1) 
{
	DC_REAL*** ninv;
	int ninv_x, ninv_y, ninv_z;
	int ninv_cx, ninv_cy, ninv_cz;
	DC_REAL ninv_size;
	int i, j, k, l, m, n, d;
	size_t nodeNum = mesh_x * mesh_y * mesh_z;
	DC_REAL corr;
	//DC_REAL corr_min = 10;
	//DC_REAL corr_max = -10;
	//DC_REAL corr_mean = 0;
	DC_REAL*** vd1_patch;
	DC_REAL*** vd2_patch;
	//
	int vd1_xm = vd1_x;
	int vd1_ym = vd1_y;
	int vd1_zm = vd1_z;
	int vd2_xm = vd2_x;
	int vd2_ym = vd2_y;
	int vd2_zm = vd2_z;
	int vd1_xo = 0;
	int vd1_yo = 0;
	int vd1_zo = 0;
	int vd2_xo = 0;
	int vd2_yo = 0;
	int vd2_zo = 0;

	if (dc_weight == 0) {
#if 0
		ninv_x = mesh_ex * 4 * ninv_s;
		ninv_y = mesh_ey * 4 * ninv_s;
		ninv_z = mesh_ez * 4 * ninv_s;
#endif
#if 1
		ninv_x = 1 * 4 * ninv_s;
		ninv_y = 1 * 4 * ninv_s;
		ninv_z = 1 * 4 * ninv_s;
#endif
#if 0
		ninv_x = 2 * 4 * ninv_s;
		ninv_y = 2 * 4 * ninv_s;
		ninv_z = 2 * 4 * ninv_s;
#endif
		ninv_cx = ninv_x / 2;
		ninv_cy = ninv_y / 2;
		ninv_cz = ninv_z / 2;
		//
		ninv_size = (float)(ninv_x * ninv_y * ninv_z);
		ninv = (DC_REAL***)malloc(ninv_z * sizeof(DC_REAL**));
		for (k = 0; k < ninv_z; k++) {
			ninv[k] = (DC_REAL**)malloc(ninv_y * sizeof(DC_REAL*));
			for (j = 0; j < ninv_y; j++) {
				ninv[k][j] = (DC_REAL*)malloc(ninv_x * sizeof(DC_REAL));
			}
		}
		get_ninv(ninv, ninv_x / 4, ninv_y / 4, ninv_z / 4);
	} else {
#if 0
		ninv_x = mesh_ex * 2 * ninv_s;
		ninv_y = mesh_ey * 2 * ninv_s;
		ninv_z = mesh_ez * 2 * ninv_s;
#endif
#if 1
		ninv_x = 1 * 2 * ninv_s;
		ninv_y = 1 * 2 * ninv_s;
		ninv_z = 1 * 2 * ninv_s;
#endif
#if 0
		ninv_x = 2 * 2 * ninv_s;
		ninv_y = 2 * 2 * ninv_s;
		ninv_z = 2 * 2 * ninv_s;
#endif
		ninv_cx = ninv_x / 2;
		ninv_cy = ninv_y / 2;
		ninv_cz = ninv_z / 2;
		//
		ninv_size = (float)(ninv_x * ninv_y * ninv_z);
		DC_REAL _ninv_size = 1.0f / ninv_size;
		ninv = (DC_REAL***)malloc(ninv_z * sizeof(DC_REAL**));
		for (k = 0; k < ninv_z; k++) {
			ninv[k] = (DC_REAL**)malloc(ninv_y * sizeof(DC_REAL*));
			for (j = 0; j < ninv_y; j++) {
				ninv[k][j] = (DC_REAL*)malloc(ninv_x * sizeof(DC_REAL));
				for (i = 0; i < ninv_x; i++) {
					ninv[k][j][i] = _ninv_size;
				}
			}
		}
	}
	/*{
		FILE* fp;
		fp = fopen("ninv.txt", "w");
		for (k = 0; k < ninv_z; k++) {
			for (j = 0; j < ninv_y; j++) {
				for (i = 0; i < ninv_x; i++) {
					fprintf(fp, "%e ", ninv[k][j][i]);
				}
				fprintf(fp, "\n");
			}
		}
		fclose(fp);
	}*/
	{
		DC_REAL ninv_sum = 0;
		for (k = 0; k < ninv_z; k++) {
			for (j = 0; j < ninv_y; j++) {
				for (i = 0; i < ninv_x; i++) {
					ninv_sum += ninv[k][j][i];
				}
			}
		}
		TRACE("ninv_sum = %f\n", ninv_sum);
	}

	vd1_patch = (DC_REAL***)malloc(ninv_z * sizeof(DC_REAL**));
	vd2_patch = (DC_REAL***)malloc(ninv_z * sizeof(DC_REAL**));
	for (k = 0; k < ninv_z; k++) {
		vd1_patch[k] = (DC_REAL**)malloc(ninv_y * sizeof(DC_REAL*));
		vd2_patch[k] = (DC_REAL**)malloc(ninv_y * sizeof(DC_REAL*));
		for (j = 0; j < ninv_y; j++) {
			vd1_patch[k][j] = (DC_REAL*)malloc(ninv_x * sizeof(DC_REAL));
			vd2_patch[k][j] = (DC_REAL*)malloc(ninv_x * sizeof(DC_REAL));
		}
	}

	for (n = 0; n < mesh_z; n++) {
		TRACE("processing %d / %d total z\n", n, mesh_z);
		for (m = 0; m < mesh_y; m++) {
			for (l = 0; l < mesh_x; l++) {
				DC_REAL x20, y20, z20;
				DC_REAL x2, y2, z2;
				DC_REAL x10, y10, z10;
				DC_REAL x1, y1, z1;
				DC_REAL dx, dy, dz;
				DC_REAL di, vd1, vd2;
				int ix1, iy1, iz1;
				int ix2, iy2, iz2;
				DC_REAL fx, fy, fz, fx1, fy1, fz1;
				DC_REAL vd1_m, vd1_c, vd2_m, vd2_c;
				//
				x10 = X1[n][m][l][0] + dX1[n][m][l][0] + vd1_xo - ninv_cx;
				y10 = Y1[n][m][l][0] + dY1[n][m][l][0] + vd1_yo - ninv_cy;
				z10 = Z1[n][m][l][0] + dZ1[n][m][l][0] + vd1_zo - ninv_cz;
				//
				if (dc_skip_back == 1) {
					x1 = x10 + ninv_cx;
					y1 = y10 + ninv_cy;
					z1 = z10 + ninv_cz;
					if ((x1 <= 0) || (x1 >= vd1_x-1) || (y1 <= 0) || (y1 >= vd1_y-1) || (z1 <= 0) || (z1 >= vd1_z-1)) {
						for (d = 0; d < num_d; d++) {
							dcv[n][m][l][d] = 0;
						}
						continue;
					} else {
						if (vdata1[(int)z1][(int)y1][(int)x1][0] <= dc_back_color) {
							for (d = 0; d < num_d; d++) {
								dcv[n][m][l][d] = 0;
							}
							continue;
						}
					}
				}
				//
				vd1_m = 0;
				for (k = 0; k < ninv_z; k++) {
					for (j = 0; j < ninv_y; j++) {
						for (i = 0; i < ninv_x; i++) {
							x1 = x10 + i;
							y1 = y10 + j;
							z1 = z10 + k;
							if ((x1 <= 0) || (x1 >= vd1_x-1) || (y1 <= 0) || (y1 >= vd1_y-1) || (z1 <= 0) || (z1 >= vd1_z-1)) {
								vd1 = 0;
							} else {
								ix1 = (int)x1;
								iy1 = (int)y1;
								iz1 = (int)z1;
								fx = x1 - ix1;
								fy = y1 - iy1;
								fz = z1 - iz1;
								if (fx == 0 && fy == 0 && fz == 0) {
									vd1 = vdata1[iz1  ][iy1  ][ix1  ][0];
								} else {
									fx1 = 1.0f - fx;
									fy1 = 1.0f - fy;
									fz1 = 1.0f - fz;
									vd1  = fx1*fy1*fz1*vdata1[iz1  ][iy1  ][ix1  ][0];
									vd1 += fx *fy1*fz1*vdata1[iz1  ][iy1  ][ix1+1][0];
									vd1 += fx1*fy *fz1*vdata1[iz1  ][iy1+1][ix1  ][0];
									vd1 += fx1*fy1*fz *vdata1[iz1+1][iy1  ][ix1  ][0];
									vd1 += fx *fy *fz1*vdata1[iz1  ][iy1+1][ix1+1][0];
									vd1 += fx *fy1*fz *vdata1[iz1+1][iy1  ][ix1+1][0];
									vd1 += fx1*fy *fz *vdata1[iz1+1][iy1+1][ix1  ][0];
									vd1 += fx *fy *fz *vdata1[iz1+1][iy1+1][ix1+1][0];
								}
							}
							vd1_patch[k][j][i] = vd1;
							vd1_m += vd1;
						}
					}
				}
#ifdef DUMP_DC_PATCHES
				{
					FILE* fp;
					char str_dc_p[1024];
					BYTE vd1_pb[1024];
					sprintf(str_dc_p, str_dc_pat_h, k, l);
					fp = fopen(str_dc_p, "wb");
					for (k = 0; k < ninv_z; k++) {
						for (j = 0; j < ninv_y; j++) {
							for (i = 0; i < ninv_x; i++) {
								vd1_pb[i] = CLAMP(vd2_patch[k][j][i], 0, 255);
							}
							fwrite(vd2_pb, 1, ninv_x, fp);
						}
					}
					fclose(fp);
				}
#endif
				vd1_m /= ninv_size;
				vd1_c = 0;
				for (k = 0; k < ninv_z; k++) {
					for (j = 0; j < ninv_y; j++) {
						for (i = 0; i < ninv_x; i++) {
							di = vd1_patch[k][j][i] - vd1_m;
							vd1_patch[k][j][i] = di;
							vd1_c += ninv[k][j][i] * di * di;
						}
					}
				}
				vd1_c = sqrt(vd1_c);
				//
				x20 = X2[n][m][l][0] + dX2[n][m][l][0] + vd2_xo - ninv_cx;
				y20 = Y2[n][m][l][0] + dY2[n][m][l][0] + vd2_yo - ninv_cy;
				z20 = Z2[n][m][l][0] + dZ2[n][m][l][0] + vd2_zo - ninv_cz;
				//
				for (d = 0; d < num_d; d++) {
					dx = disp_x[d];
					dy = disp_y[d];
					dz = disp_z[d];
					//
					vd2_m = 0;
					for (k = 0; k < ninv_z; k++) {
						for (j = 0; j < ninv_y; j++) {
							for (i = 0; i < ninv_x; i++) {
								x2 = x20 + i + dx;
								y2 = y20 + j + dy;
								z2 = z20 + k + dz;
								if ((x2 <= 0) || (x2 >= vd2_x-1) || (y2 <= 0) || (y2 >= vd2_y-1) || (z2 <= 0) || (z2 >= vd2_z-1)) {
									vd2 = 0;
								} else {
									ix2 = (int)x2;
									iy2 = (int)y2;
									iz2 = (int)z2;
									fx = x2 - ix2;
									fy = y2 - iy2;
									fz = z2 - iz2;
									if (fx == 0 && fy == 0 && fz == 0) {
										vd2 = vdata2[iz2  ][iy2  ][ix2  ][0];
									} else {
										fx1 = 1.0f - fx;
										fy1 = 1.0f - fy;
										fz1 = 1.0f - fz;
										vd2  = fx1*fy1*fz1*vdata2[iz2  ][iy2  ][ix2  ][0];
										vd2 += fx *fy1*fz1*vdata2[iz2  ][iy2  ][ix2+1][0];
										vd2 += fx1*fy *fz1*vdata2[iz2  ][iy2+1][ix2  ][0];
										vd2 += fx1*fy1*fz *vdata2[iz2+1][iy2  ][ix2  ][0];
										vd2 += fx *fy *fz1*vdata2[iz2  ][iy2+1][ix2+1][0];
										vd2 += fx *fy1*fz *vdata2[iz2+1][iy2  ][ix2+1][0];
										vd2 += fx1*fy *fz *vdata2[iz2+1][iy2+1][ix2  ][0];
										vd2 += fx *fy *fz *vdata2[iz2+1][iy2+1][ix2+1][0];
									}
								}
								vd2_patch[k][j][i] = vd2;
								vd2_m += vd2;
							}
						}
					}
					vd2_m /= ninv_size;
					vd2_c = 0;
					for (k = 0; k < ninv_z; k++) {
						for (j = 0; j < ninv_y; j++) {
							for (i = 0; i < ninv_x; i++) {
								di = vd2_patch[k][j][i] - vd2_m;
								vd2_patch[k][j][i] = di;
								vd2_c += ninv[k][j][i] * di * di;
							}
						}
					}
					vd2_c = sqrt(vd2_c);
					//
					corr = 0;
					for (k = 0; k < ninv_z; k++) {
						for (j = 0; j < ninv_y; j++) {
							for (i = 0; i < ninv_x; i++) {
								di = vd1_patch[k][j][i] * vd2_patch[k][j][i];
								corr += ninv[k][j][i] * di;
							}
						}
					}
					if ((vd1_c != 0) && (vd2_c != 0)) {
						corr /= vd1_c * vd2_c;
					} else {
						corr = 0;
					}
					//dcv[n][m][l][d] = 1.0 - corr;
					dcv[n][m][l][d] = 255 * 128 * (1.0f - corr);
				}
				//corr_mean += dcv[n][m][l][num_d/2];
			}
		}
	}

	for (k = 0; k < ninv_z; k++) {
		for (j = 0; j < ninv_y; j++) {
			free(ninv[k][j]);
		}
		free(ninv[k]);
	}
	free(ninv);
	//
	for (k = 0; k < ninv_z; k++) {
		for (j = 0; j < ninv_y; j++) {
			free(vd1_patch[k][j]);
			free(vd2_patch[k][j]);
		}
		free(vd1_patch[k]);
		free(vd2_patch[k]);
	}
	free(vd1_patch);
	free(vd2_patch);
}

template <class T>
void ComputeDataCost3D_CC(T**** vdata1, int vd1_x, int vd1_y, int vd1_z, int vd1_s, T**** vdata2, int vd2_x, int vd2_y, int vd2_z, int vd2_s,
	REALV**** X1, REALV**** Y1, REALV**** Z1, REALV**** dX1, REALV**** dY1, REALV**** dZ1, REALV**** X2, REALV**** Y2, REALV**** Z2, REALV**** dX2, REALV**** dY2, REALV**** dZ2,
	int mesh_x, int mesh_y, int mesh_z, int mesh_ex, int mesh_ey, int mesh_ez,
	REALV* disp_x, REALV* disp_y, REALV* disp_z, int num_d, REALV**** dcv,
	int dc_skip_back, T dc_back_color, int radius = 4) 
{
	int ninv_x, ninv_y, ninv_z;
	int ninv_cx, ninv_cy, ninv_cz;
	int i, j, k, l, m, n, d;
	double*** vd1_patch;
	double max_corr, min_corr;

	ninv_x = 2 * radius + 1;
	ninv_y = 2 * radius + 1;
	ninv_z = 2 * radius + 1;
	//
	ninv_cx = ninv_x / 2;
	ninv_cy = ninv_y / 2;
	ninv_cz = ninv_z / 2;

	vd1_patch = (double***)malloc(ninv_z * sizeof(double**));
	for (k = 0; k < ninv_z; k++) {
		vd1_patch[k] = (double**)malloc(ninv_y * sizeof(double*));
		for (j = 0; j < ninv_y; j++) {
			vd1_patch[k][j] = (double*)malloc(ninv_x * sizeof(double));
		}
	}

	max_corr = -10000;
	min_corr = 10000;
	for (n = 0; n < mesh_z; n++) {
		TRACE("processing %d / %d total z\n", n, mesh_z);
		for (m = 0; m < mesh_y; m++) {
			for (l = 0; l < mesh_x; l++) {
				float x10, y10, z10;
				float x20, y20, z20;
				float x1, y1, z1;
				float x2, y2, z2;
				float dx, dy, dz;
				int ix1, iy1, iz1;
				int ix2, iy2, iz2;
				double vd1, vd2;
				double fx, fy, fz, fx1, fy1, fz1;
				double fixedMean, movingMean;
				double suma2, sumb2, suma, sumb, sumab, count;
				double sff, smm, sfm;
				double corr;
				double corr_max = 1.0;
				//
				x10 = X1[n][m][l][0] + dX1[n][m][l][0] - ninv_cx;
				y10 = Y1[n][m][l][0] + dY1[n][m][l][0] - ninv_cy;
				z10 = Z1[n][m][l][0] + dZ1[n][m][l][0] - ninv_cz;
				//
				if (dc_skip_back == 1) {
					x1 = x10 + ninv_cx;
					y1 = y10 + ninv_cy;
					z1 = z10 + ninv_cz;
					if ((x1 <= 0) || (x1 >= vd1_x-1) || (y1 <= 0) || (y1 >= vd1_y-1) || (z1 <= 0) || (z1 >= vd1_z-1)) {
						for (d = 0; d < num_d; d++) {
							dcv[n][m][l][d] = 0;
						}
						continue;
					} else {
						if (vdata1[(int)z1][(int)y1][(int)x1][0] <= dc_back_color) {
							for (d = 0; d < num_d; d++) {
								dcv[n][m][l][d] = 0;
							}
							continue;
						}
					}
				}
				//
				for (k = 0; k < ninv_z; k++) {
					for (j = 0; j < ninv_y; j++) {
						for (i = 0; i < ninv_x; i++) {
							x1 = x10 + i;
							y1 = y10 + j;
							z1 = z10 + k;
							if ((x1 <= 0) || (x1 >= vd1_x-1) || (y1 <= 0) || (y1 >= vd1_y-1) || (z1 <= 0) || (z1 >= vd1_z-1)) {
								//vd1_patch[k][j][i] = -1;
								vd1_patch[k][j][i] = 0;
							} else {
								ix1 = (int)x1;
								iy1 = (int)y1;
								iz1 = (int)z1;
								fx = x1 - ix1;
								fy = y1 - iy1;
								fz = z1 - iz1;
								if (fx == 0 && fy == 0 && fz == 0) {
									vd1 = vdata1[iz1  ][iy1  ][ix1  ][0];
								} else {
									fx1 = 1.0f - fx;
									fy1 = 1.0f - fy;
									fz1 = 1.0f - fz;
									vd1  = fx1*fy1*fz1*vdata1[iz1  ][iy1  ][ix1  ][0];
									vd1 += fx *fy1*fz1*vdata1[iz1  ][iy1  ][ix1+1][0];
									vd1 += fx1*fy *fz1*vdata1[iz1  ][iy1+1][ix1  ][0];
									vd1 += fx1*fy1*fz *vdata1[iz1+1][iy1  ][ix1  ][0];
									vd1 += fx *fy *fz1*vdata1[iz1  ][iy1+1][ix1+1][0];
									vd1 += fx *fy1*fz *vdata1[iz1+1][iy1  ][ix1+1][0];
									vd1 += fx1*fy *fz *vdata1[iz1+1][iy1+1][ix1  ][0];
									vd1 += fx *fy *fz *vdata1[iz1+1][iy1+1][ix1+1][0];
								}
								vd1_patch[k][j][i] = vd1;
							}
						}
					}
				}
				//
				x20 = X2[n][m][l][0] + dX2[n][m][l][0] - ninv_cx;
				y20 = Y2[n][m][l][0] + dY2[n][m][l][0] - ninv_cy;
				z20 = Z2[n][m][l][0] + dZ2[n][m][l][0] - ninv_cz;
				//
				for (d = 0; d < num_d; d++) {
					dx = disp_x[d];
					dy = disp_y[d];
					dz = disp_z[d];
					//
					count = 0;
					suma2 = suma = 0;
					sumb2 = sumb = 0;
					sumab = 0;
					for (k = 0; k < ninv_z; k++) {
						for (j = 0; j < ninv_y; j++) {
							for (i = 0; i < ninv_x; i++) {
								vd1 = vd1_patch[k][j][i];
								//if (vd1 < 0) {
								//	continue;
								//}
								//
								x2 = x20 + i + dx;
								y2 = y20 + j + dy;
								z2 = z20 + k + dz;
								if ((x2 <= 0) || (x2 >= vd2_x-1) || (y2 <= 0) || (y2 >= vd2_y-1) || (z2 <= 0) || (z2 >= vd2_z-1)) {
									vd2 = 0;
								} else {
									ix2 = (int)x2;
									iy2 = (int)y2;
									iz2 = (int)z2;
									fx = x2 - ix2;
									fy = y2 - iy2;
									fz = z2 - iz2;
									if (fx == 0 && fy == 0 && fz == 0) {
										vd2 = vdata2[iz2  ][iy2  ][ix2  ][0];
									} else {
										fx1 = 1.0f - fx;
										fy1 = 1.0f - fy;
										fz1 = 1.0f - fz;
										vd2  = fx1*fy1*fz1*vdata2[iz2  ][iy2  ][ix2  ][0];
										vd2 += fx *fy1*fz1*vdata2[iz2  ][iy2  ][ix2+1][0];
										vd2 += fx1*fy *fz1*vdata2[iz2  ][iy2+1][ix2  ][0];
										vd2 += fx1*fy1*fz *vdata2[iz2+1][iy2  ][ix2  ][0];
										vd2 += fx *fy *fz1*vdata2[iz2  ][iy2+1][ix2+1][0];
										vd2 += fx *fy1*fz *vdata2[iz2+1][iy2  ][ix2+1][0];
										vd2 += fx1*fy *fz *vdata2[iz2+1][iy2+1][ix2  ][0];
										vd2 += fx *fy *fz *vdata2[iz2+1][iy2+1][ix2+1][0];
									}
								}
								//
								suma  += vd1;
								suma2 += vd1 * vd1;
								sumb  += vd2;
								sumb2 += vd2 * vd2;
								sumab += vd1 * vd2;
								count += 1;
							}
						}
					}
					//
					if (count > 0) {
						fixedMean  = suma / count;
						movingMean = sumb / count;
						sff = suma2 - fixedMean*suma  - fixedMean*suma  + count*fixedMean*fixedMean;
						smm = sumb2 - movingMean*sumb - movingMean*sumb + count*movingMean*movingMean;
						sfm = sumab - movingMean*suma - fixedMean*sumb  + count*movingMean*fixedMean;
						//
						//if ((sff > 0) && (smm > 0)) {
						if ((sff > 1e-1) && (smm > 1e-1)) {
#ifdef USE_CC_NCC
							corr = sfm / sqrt(sff * smm);
#else
							corr = (sfm * sfm) / (sff * smm);
#endif

							if (corr > max_corr) {
								max_corr = corr;
							}
							if (corr < min_corr) {
								min_corr = corr;
							}
							//*
							if (corr > corr_max) {
								corr = corr_max;
							}
							//*/
						} else {
							corr = 0;
						}
					} else {
						corr = 0;
					}
					//
					dcv[n][m][l][d] = (float)(255 * 128 * (corr_max - corr));
				}
				//corr_mean += dcv[n][m][l][num_d/2];
			}
		}
	}

	TRACE("\nmin_corr: %f, max_corr = %f\n", min_corr, max_corr);

	for (k = 0; k < ninv_z; k++) {
		for (j = 0; j < ninv_y; j++) {
			free(vd1_patch[k][j]);
		}
		free(vd1_patch[k]);
	}
	free(vd1_patch);
}

template <class T>
void ComputeDataCost3D_CC_Fast(T**** vdata1, int vd1_x, int vd1_y, int vd1_z, int vd1_s, T**** vdata2, int vd2_x, int vd2_y, int vd2_z, int vd2_s,
	REALV**** XC, REALV**** YC, REALV**** ZC,
	int mesh_x, int mesh_y, int mesh_z, int mesh_ex, int mesh_ey, int mesh_ez,
	REALV* disp_x, REALV* disp_y, REALV* disp_z, int num_d, REALV**** dcv,
	int dc_skip_back, T dc_back_color, int radius = 4) 
{
	int ninv_x, ninv_y, ninv_z;
	int ninv_cx, ninv_cy, ninv_cz;
	int ninv_size;
	int i, j, k, l, m, n, d;
	double max_corr, min_corr;
	REALV disp_x_max, disp_x_min, disp_y_max, disp_y_min, disp_z_max, disp_z_min;
	int lx, ly, lz, rx, ry, rz;
	DVolume vd1t, vd2t;
	int vdt_x, vdt_y, vdt_z;

	ninv_x = 2 * radius + 1;
	ninv_y = 2 * radius + 1;
	ninv_z = 2 * radius + 1;
	ninv_size = ninv_x * ninv_y * ninv_z;
	//
	ninv_cx = ninv_x / 2;
	ninv_cy = ninv_y / 2;
	ninv_cz = ninv_z / 2;

	disp_x_max = -10000; disp_x_min = 10000;
	disp_y_max = -10000; disp_y_min = 10000;
	disp_z_max = -10000; disp_z_min = 10000;
	for (d = 0; d < num_d; d++) {
		if (disp_x[d] > disp_x_max) { disp_x_max = disp_x[d]; }
		if (disp_x[d] < disp_x_min) { disp_x_min = disp_x[d]; }
		if (disp_y[d] > disp_y_max) { disp_y_max = disp_x[d]; }
		if (disp_y[d] < disp_y_min) { disp_y_min = disp_x[d]; }
		if (disp_z[d] > disp_z_max) { disp_z_max = disp_x[d]; }
		if (disp_z[d] < disp_z_min) { disp_z_min = disp_x[d]; }
	}
	
	lx = (int)(fabs(disp_x_min) + 0.999999) + ninv_cx;
	rx = (int)(fabs(disp_x_max) + 0.999999) + ninv_cx+1;
	ly = (int)(fabs(disp_y_min) + 0.999999) + ninv_cy;
	ry = (int)(fabs(disp_y_max) + 0.999999) + ninv_cy+1;
	lz = (int)(fabs(disp_z_min) + 0.999999) + ninv_cz;
	rz = (int)(fabs(disp_z_max) + 0.999999) + ninv_cz+1;

	vdt_x = max(vd1_x+lx+rx, vd2_x+lx+rx);
	vdt_y = max(vd1_y+ly+ry, vd2_y+ly+ry);
	vdt_z = max(vd1_z+lz+rz, vd2_z+lz+rz);

	vd1t.allocate(vdt_x, vdt_y, vdt_z);
	vd2t.allocate(vdt_x, vdt_y, vdt_z);
	
	max_corr = -10000;
	min_corr = 10000;
	for (d = 0; d < num_d; d++) {
		float xc, yc, zc;
		float x1, y1, z1;
		float x2, y2, z2;
		float dx, dy, dz;
		int ix1, iy1, iz1;
		int ix2, iy2, iz2;
		int ixc, iyc, izc;
		int ix, iy, iz;
		double vd1, vd2;
		double fx, fy, fz, fx1, fy1, fz1;
		double fixedMean, movingMean;
		double suma2, sumb2, suma, sumb, sumab, count;
		double sff, smm, sfm;
		double corr;
		double corr_max = 1.0;

		if (d % (int)(num_d * 0.1) == 0) {
			TRACE("processing %d / %d total d\n", d, num_d);
		}

		dx = disp_x[d];
		dy = disp_y[d];
		dz = disp_z[d];

		// translate images
		for (k = 0; k < vdt_z; k++) {
			for (j = 0; j < vdt_y; j++) {
				for (i = 0; i < vdt_x; i++) {
					ix1 = i - lx;
					iy1 = j - ly;
					iz1 = k - lz;
					if ((ix1 < 0) || (ix1 > vd1_x-1) || (iy1 < 0) || (iy1 > vd1_y-1) || (iz1 < 0) || (iz1 > vd1_z-1)) {
						vd1t.m_pData[k][j][i][0] = 0;
					} else {
						vd1t.m_pData[k][j][i][0] = vdata1[iz1][iy1][ix1][0];
					}

					x2 = i - (lx - dx);
					y2 = j - (ly - dy);
					z2 = k - (lz - dz);
					ix2 = (int)x2;
					iy2 = (int)y2;
					iz2 = (int)z2;
					fx = x2 - ix2;
					fy = y2 - iy2;
					fz = z2 - iz2;
					if (fx == 0 && fy == 0 && fz == 0) {
						if ((ix2 < 0) || (ix2 > vd2_x-1) || (iy2 < 0) || (iy2 > vd2_y-1) || (iz2 < 0) || (iz2 > vd2_z-1)) {
							vd2t.m_pData[k][j][i][0] = 0;
						} else {
							vd2t.m_pData[k][j][i][0] = vdata2[iz2][iy2][ix2][0];
						}
					} else {
						if ((ix2 < 0) || (ix2 >= vd2_x-1) || (iy2 < 0) || (iy2 >= vd2_y-1) || (iz2 < 0) || (iz2 >= vd2_z-1)) {
							vd2t.m_pData[k][j][i][0] = 0;
						} else {
							fx1 = 1.0f - fx;
							fy1 = 1.0f - fy;
							fz1 = 1.0f - fz;
							vd2  = fx1*fy1*fz1*vdata2[iz2  ][iy2  ][ix2  ][0];
							vd2 += fx *fy1*fz1*vdata2[iz2  ][iy2  ][ix2+1][0];
							vd2 += fx1*fy *fz1*vdata2[iz2  ][iy2+1][ix2  ][0];
							vd2 += fx1*fy1*fz *vdata2[iz2+1][iy2  ][ix2  ][0];
							vd2 += fx *fy *fz1*vdata2[iz2  ][iy2+1][ix2+1][0];
							vd2 += fx *fy1*fz *vdata2[iz2+1][iy2  ][ix2+1][0];
							vd2 += fx1*fy *fz *vdata2[iz2+1][iy2+1][ix2  ][0];
							vd2 += fx *fy *fz *vdata2[iz2+1][iy2+1][ix2+1][0];
							vd2t.m_pData[k][j][i][0] = vd2;
						}
					}
				}
			}
		}
		//vd1t.save("vd1t.nii.gz", 1);
		//vd2t.save("vd2t.nii.gz", 1);

		for (n = 0; n < mesh_z; n++) {
			for (m = 0; m < mesh_y; m++) {
				for (l = 0; l < mesh_x; l++) {
					xc = XC[n][m][l][0];
					yc = YC[n][m][l][0];
					zc = ZC[n][m][l][0];
					//
					if (dc_skip_back == 1) {
						if ((xc < 0) || (xc > vd1_x-1) || (yc < 0) || (yc > vd1_y-1) || (zc < 0) || (zc > vd1_z-1)) {
							dcv[n][m][l][d] = 0;
							continue;
						} else {
							//*
							if (vdata1[(int)zc][(int)yc][(int)xc][0] <= dc_back_color) {
								dcv[n][m][l][d] = 0;
								continue;
							}
							//*/
						}
					}
					//
					ixc = (int)(xc - ninv_cx + lx);
					iyc = (int)(yc - ninv_cy + ly);
					izc = (int)(zc - ninv_cz + lz);
					//
					//count = 0;
					suma2 = suma = 0;
					sumb2 = sumb = 0;
					sumab = 0;
					for (k = 0; k < ninv_z; k++) {
						for (j = 0; j < ninv_y; j++) {
							for (i = 0; i < ninv_x; i++) {
								ix = ixc + i;
								iy = iyc + j;
								iz = izc + k;
								//
								vd1 = vd1t.m_pData[iz][iy][ix][0];
								vd2 = vd2t.m_pData[iz][iy][ix][0];
								//
								suma  += vd1;
								suma2 += vd1 * vd1;
								sumb  += vd2;
								sumb2 += vd2 * vd2;
								sumab += vd1 * vd2;
								//count += 1;
							}
						}
					}
					count = ninv_size;
					//
					if (count > 0) {
						fixedMean  = suma / count;
						movingMean = sumb / count;
						sff = suma2 -  fixedMean*suma -  fixedMean*suma + count* fixedMean* fixedMean;
						smm = sumb2 - movingMean*sumb - movingMean*sumb + count*movingMean*movingMean;
						sfm = sumab - movingMean*suma -  fixedMean*sumb + count*movingMean* fixedMean;
						//
						//if ((sff > 0) && (smm > 0)) {
						//if ((sff > 1e-1) && (smm > 1e-1)) {
						if (sff*smm > 1.e-5) {
#ifdef USE_CC_NCC
							corr = sfm / sqrt(sff * smm);
#else
							corr = (sfm * sfm) / (sff * smm);
#endif

							if (corr > max_corr) {
								max_corr = corr;
							}
							if (corr < min_corr) {
								min_corr = corr;
							}
							//*
							if (corr > corr_max) {
								corr = corr_max;
							}
							//*/
						} else {
							corr = 0;
						}
					} else {
						corr = 0;
					}
					//
					dcv[n][m][l][d] = (float)(255 * 128 * (corr_max - corr));
				}
			}
		}
	}

	TRACE("\nmin_corr: %f, max_corr = %f\n", min_corr, max_corr);
}

//#define TEST_GRADIENT
template <class T>
void ComputeDataCost3D_CC_ApplyWeight(T**** vdata1, int vd1_x, int vd1_y, int vd1_z, int vd1_s, T**** vdata2, int vd2_x, int vd2_y, int vd2_z, int vd2_s,
	REALV**** XC, REALV**** YC, REALV**** ZC,
	int mesh_x, int mesh_y, int mesh_z, int mesh_ex, int mesh_ey, int mesh_ez,
	REALV* disp_x, REALV* disp_y, REALV* disp_z, int num_d, REALV**** dcv,
	int dc_skip_back, T dc_back_color, int radius = 4) 
{
	int ninv_x, ninv_y, ninv_z;
	int ninv_cx, ninv_cy, ninv_cz;
	int ninv_size;
	int i, j, k, l, m, n, d;
	double max_w, min_w, max_w_1;
	int lx, ly, lz, rx, ry, rz;
	DVolume vd1t, vd2t;
	DVolume vd1gx, vd1gy, vd1gz;
	DVolume vdw, vdwx, vdwy, vdwz, vdwx_g, vdwy_g, vdwz_g;
	int vdt_x, vdt_y, vdt_z;

	ninv_x = 2 * radius + 1;
	ninv_y = 2 * radius + 1;
	ninv_z = 2 * radius + 1;
	ninv_size = ninv_x * ninv_y * ninv_z;
	//
	ninv_cx = ninv_x / 2;
	ninv_cy = ninv_y / 2;
	ninv_cz = ninv_z / 2;

	lx = ninv_cx;
	rx = ninv_cx+1;
	ly = ninv_cy;
	ry = ninv_cy+1;
	lz = ninv_cz;
	rz = ninv_cz+1;

	vdt_x = max(vd1_x+lx+rx, vd2_x+lx+rx);
	vdt_y = max(vd1_y+ly+ry, vd2_y+ly+ry);
	vdt_z = max(vd1_z+lz+rz, vd2_z+lz+rz);

	vd1t.allocate(vdt_x, vdt_y, vdt_z);
	vd2t.allocate(vdt_x, vdt_y, vdt_z);
	vd1gx.allocate(vdt_x, vdt_y, vdt_z);
	vd1gy.allocate(vdt_x, vdt_y, vdt_z);
	vd1gz.allocate(vdt_x, vdt_y, vdt_z);
	//
	vdwx.allocate(vd1_x, vd1_y, vd1_z);
	vdwy.allocate(vd1_x, vd1_y, vd1_z);
	vdwz.allocate(vd1_x, vd1_y, vd1_z);
	
	{
		float xc, yc, zc;
		int ixc, iyc, izc;
		int ix, iy, iz;
		double vd1, vd2;
		double fixedMean, movingMean;
		double suma2, sumb2, suma, sumb, sumab, count;
		double sff, smm, sfm;
		double w;
		double wx, wy, wz;
		double dx, dy, dz;

		// translate images
		for (k = 0; k < vdt_z; k++) {
			for (j = 0; j < vdt_y; j++) {
				for (i = 0; i < vdt_x; i++) {
					ixc = i - lx;
					iyc = j - ly;
					izc = k - lz;
					if ((ixc <= 0) || (ixc >= vd1_x-1) || (iyc <= 0) || (iyc >= vd1_y-1) || (izc <= 0) || (izc >= vd1_z-1)) {
						vd1 = 0;
						vd2 = 0;
					} else {
						vd1 = vdata1[izc][iyc][ixc][0];
						vd2 = vdata2[izc][iyc][ixc][0];
					}
					vd1t.m_pData[k][j][i][0] = vd1;
					vd2t.m_pData[k][j][i][0] = vd2;
				}
			}
		}
		
		vd1t.GetGradient(vd1gx, vd1gy, vd1gz);

#ifdef TEST_GRADIENT
		vd1t.save("vd1t.nii.gz", 1);
		vd2t.save("vd2t.nii.gz", 1);
		SaveMHDData(NULL, "vd1g.mhd", vd1gx.m_pData, vd1gy.m_pData, vd1gz.m_pData, vdt_x, vdt_y, vdt_z, 1, 1, 1, 0, 0, 0);
#endif

		for (n = 0; n < vd1_z; n++) {
			for (m = 0; m < vd1_y; m++) {
				for (l = 0; l < vd1_x; l++) {
					xc = l;
					yc = m;
					zc = n;
					//
					if (dc_skip_back == 1) {
						if ((xc <= 0) || (xc >= vd1_x-1) || (yc <= 0) || (yc >= vd1_y-1) || (zc <= 0) || (zc >= vd1_z-1)) {
							continue;
						} else {
							//*
							if (vdata1[(int)zc][(int)yc][(int)xc][0] <= dc_back_color) {
								continue;
							}
							//*/
						}
					}
					//
					ixc = (int)(xc - ninv_cx + lx);
					iyc = (int)(yc - ninv_cy + ly);
					izc = (int)(zc - ninv_cz + lz);
					//
					//count = 0;
					suma2 = suma = 0;
					sumb2 = sumb = 0;
					sumab = 0;
					for (k = 0; k < ninv_z; k++) {
						for (j = 0; j < ninv_y; j++) {
							for (i = 0; i < ninv_x; i++) {
								ix = ixc + i;
								iy = iyc + j;
								iz = izc + k;
								//
								vd1 = vd1t.m_pData[iz][iy][ix][0];
								vd2 = vd2t.m_pData[iz][iy][ix][0];
								//
								suma  += vd1;
								suma2 += vd1 * vd1;
								sumb  += vd2;
								sumb2 += vd2 * vd2;
								sumab += vd1 * vd2;
								//count += 1;
							}
						}
					}
					count = ninv_size;
					//
					if (count > 0) {
						fixedMean  = suma / count;
						movingMean = sumb / count;
						sff = suma2 -  fixedMean*suma -  fixedMean*suma + count* fixedMean* fixedMean;
						smm = sumb2 - movingMean*sumb - movingMean*sumb + count*movingMean*movingMean;
						sfm = sumab - movingMean*suma -  fixedMean*sumb + count*movingMean* fixedMean;
						//
						//if ((sff > 0) && (smm > 0)) {
						//if ((sff > 1e-1) && (smm > 1e-1)) {
						if (sff*smm > 1.e-5) {
							double Ii, Ji;
							double gIx, gIy, gIz;
							ixc = (int)(xc + lx);
							iyc = (int)(yc + ly);
							izc = (int)(zc + lz);

							Ii = vd1t.m_pData[izc][iyc][ixc][0] - fixedMean;
							Ji = vd2t.m_pData[izc][iyc][ixc][0] - movingMean;
							gIx = vd1gx.m_pData[izc][iyc][ixc][0];
							gIy = vd1gy.m_pData[izc][iyc][ixc][0];
							gIz = vd1gz.m_pData[izc][iyc][ixc][0];

#ifdef USE_CC_NCC
							w = - 1.0 / sqrt(sff*smm) * (Ji - sfm/sff * Ii);
#else
							w = - 2.0 * sfm / (sff*smm) * (Ji - sfm/sff * Ii);
#endif
							wx = w * gIx;
							wy = w * gIy;
							wz = w * gIz;
						} else {
							wx = wy = wz = 0;
						}
					} else {
						wx = wy = wz = 0;
					}
					//
					vdwx.m_pData[n][m][l][0] = wx;
					vdwy.m_pData[n][m][l][0] = wy;
					vdwz.m_pData[n][m][l][0] = wz;
				}
			}
		}

		vd1t.clear();
		vd2t.clear();
		vd1gx.clear();
		vd1gy.clear();
		vd1gz.clear();

#ifdef TEST_GRADIENT
		SaveMHDData(NULL, "vdw.mhd", vdwx.m_pData, vdwy.m_pData, vdwz.m_pData, vd1_x, vd1_y, vd1_z, 1, 1, 1, 0, 0, 0);
#endif

		vdwx_g.allocate(vd1_x, vd1_y, vd1_z);
		vdwy_g.allocate(vd1_x, vd1_y, vd1_z);
		vdwz_g.allocate(vd1_x, vd1_y, vd1_z);

		/*
		vdwx.GaussianSmoothing(vdwx_g, 1.5, 5);
		vdwy.GaussianSmoothing(vdwy_g, 1.5, 5);
		vdwz.GaussianSmoothing(vdwz_g, 1.5, 5);
		/*/
		vdwx_g = vdwx;
		vdwy_g = vdwy;
		vdwz_g = vdwz;
		//*/

		vdwx.clear();
		vdwy.clear();
		vdwz.clear();

#ifdef TEST_GRADIENT
		SaveMHDData(NULL, "vdw_g.mhd", vdwx_g.m_pData, vdwy_g.m_pData, vdwz_g.m_pData, vd1_x, vd1_y, vd1_z, 1, 1, 1, 0, 0, 0);
#endif

		vdw.allocate(vd1_x, vd1_y, vd1_z);

		max_w = -10000;
		min_w = 10000;
		for (n = 0; n < vd1_z; n++) {
			for (m = 0; m < vd1_y; m++) {
				for (l = 0; l < vd1_x; l++) {
					wx = vdwx_g.m_pData[n][m][l][0];
					wy = vdwy_g.m_pData[n][m][l][0];
					wz = vdwz_g.m_pData[n][m][l][0];
					
					w = sqrt(wx*wx + wy*wy + wz*wz);
					if (w > max_w) {
						max_w = w;
					}
					if (w < min_w) {
						min_w = w;
					}

					vdw.m_pData[n][m][l][0] = w;
				}
			}
		}
		max_w_1 = 1.0 / max_w;

		TRACE("\nmin_w: %f, max_w = %f\n", min_w, max_w);

		for (n = 0; n < vd1_z; n++) {
			for (m = 0; m < vd1_y; m++) {
				for (l = 0; l < vd1_x; l++) {
					vdwx_g.m_pData[n][m][l][0] *= max_w_1;
					vdwy_g.m_pData[n][m][l][0] *= max_w_1;
					vdwz_g.m_pData[n][m][l][0] *= max_w_1;
					vdw.m_pData[n][m][l][0] *= max_w_1;
				}
			}
		}

#ifdef TEST_GRADIENT
		SaveMHDData(NULL, "vdw_g_n.mhd", vdwx_g.m_pData, vdwy_g.m_pData, vdwz_g.m_pData, vd1_x, vd1_y, vd1_z, 1, 1, 1, 0, 0, 0);
		vdw.save("vdw_n.nii.gz", 1);
#endif

		{
			double *disp_xn, *disp_yn, *disp_zn;
			double dcv_val, dd, prod;

			disp_xn = (double*)malloc(num_d * sizeof(double));
			disp_yn = (double*)malloc(num_d * sizeof(double));
			disp_zn = (double*)malloc(num_d * sizeof(double));

			for (d = 0; d < num_d; d++) {
				dx = disp_x[d];
				dy = disp_y[d];
				dz = disp_z[d];
				dd = sqrt(dx*dx + dy*dy + dz*dz);
				if (dd > 0) {
					disp_xn[d] = dx / dd;
					disp_yn[d] = dy / dd;
					disp_zn[d] = dz / dd;
				}
			}
			
			for (n = 0; n < mesh_z; n++) {
				for (m = 0; m < mesh_y; m++) {
					for (l = 0; l < mesh_x; l++) {
						xc = XC[n][m][l][0];
						yc = YC[n][m][l][0];
						zc = ZC[n][m][l][0];
						//
						ixc = (int)(xc);
						iyc = (int)(yc);
						izc = (int)(zc);

						w = vdw.m_pData[izc][iyc][ixc][0];
						/*
						if (w > 0) {
							wx = vdwx_g.m_pData[izc][iyc][ixc][0] / w;
							wy = vdwy_g.m_pData[izc][iyc][ixc][0] / w;
							wz = vdwz_g.m_pData[izc][iyc][ixc][0] / w;

							for (d = 0; d < num_d; d++) {
								dx = disp_xn[d];
								dy = disp_yn[d];
								dz = disp_zn[d];

								prod = wx*dx + wy*dy + wz*dz;
						
								dcv[n][m][l][d] *= exp(2.0 * (1.0 - prod));// * w;
							}
						} else {
							wx = wy = wz = 0;
							for (d = 0; d < num_d; d++) {
								if (d != num_d) {
									dcv[n][m][l][d] *= exp(2.0);
								}
							}
						}
						/*/
						w = pow(w, 0.5);
						for (d = 0; d < num_d; d++) {
							dcv[n][m][l][d] *= w;
						}
						//*/
					}
				}
			}

			free(disp_xn);
			free(disp_yn);
			free(disp_zn);
		}
	}

	vdwx_g.clear();
	vdwy_g.clear();
	vdwz_g.clear();
	vdw.clear();
}

template <class T>
void ComputeDataCost3DSyD_CC(T**** vdata1, int vd1_x, int vd1_y, int vd1_z, int vd1_s, T**** vdata2, int vd2_x, int vd2_y, int vd2_z, int vd2_s,
	REALV**** XC, REALV**** YC, REALV**** ZC, REALV**** dX1, REALV**** dY1, REALV**** dZ1, REALV**** dX2, REALV**** dY2, REALV**** dZ2,
	int mesh_x, int mesh_y, int mesh_z, int mesh_ex, int mesh_ey, int mesh_ez,
	REALV* disp_x, REALV* disp_y, REALV* disp_z, int num_d, REALV**** dcv,
	int dc_skip_back, T dc_back_color, int radius = 4) 
{
	int ninv_x, ninv_y, ninv_z;
	int ninv_cx, ninv_cy, ninv_cz;
	int i, j, k, l, m, n, d;
	double max_corr, min_corr;

	ninv_x = 2 * radius + 1;
	ninv_y = 2 * radius + 1;
	ninv_z = 2 * radius + 1;
	//
	ninv_cx = ninv_x / 2;
	ninv_cy = ninv_y / 2;
	ninv_cz = ninv_z / 2;

	max_corr = -10000;
	min_corr = 10000;
	for (n = 0; n < mesh_z; n++) {
		TRACE("processing %d / %d total z\n", n, mesh_z);
		for (m = 0; m < mesh_y; m++) {
			for (l = 0; l < mesh_x; l++) {
				float xc, yc, zc;
				float x10, y10, z10;
				float x20, y20, z20;
				float x1, y1, z1;
				float x2, y2, z2;
				float dx, dy, dz;
				int ix1, iy1, iz1;
				int ix2, iy2, iz2;
				double vd1, vd2;
				double fx, fy, fz, fx1, fy1, fz1;
				double fixedMean, movingMean;
				double suma2, sumb2, suma, sumb, sumab, count;
				double sff, smm, sfm;
				double corr;
				double corr_max = 1.0;
				//
				xc = XC[n][m][l][0];
				yc = YC[n][m][l][0];
				zc = ZC[n][m][l][0];
				//
				if (dc_skip_back == 1) {
					if ((xc <= 0) || (xc >= vd1_x-1) || (yc <= 0) || (yc >= vd1_y-1) || (zc <= 0) || (zc >= vd1_z-1)) {
						for (d = 0; d < num_d; d++) {
							dcv[n][m][l][d] = 0;
						}
						continue;
					} else {
						/*
						if (vdata1[(int)z1][(int)y1][(int)x1][0] <= dc_back_color) {
							for (d = 0; d < num_d; d++) {
								dcv[n][m][l][d] = 0;
							}
							continue;
						}
						//*/
					}
				}
				//
				x10 = xc + dX1[n][m][l][0] - ninv_cx;
				y10 = yc + dY1[n][m][l][0] - ninv_cy;
				z10 = zc + dZ1[n][m][l][0] - ninv_cz;
				//
				x20 = xc + dX2[n][m][l][0] - ninv_cx;
				y20 = yc + dY2[n][m][l][0] - ninv_cy;
				z20 = zc + dZ2[n][m][l][0] - ninv_cz;
				//
				for (d = 0; d < num_d; d++) {
					dx = disp_x[d];
					dy = disp_y[d];
					dz = disp_z[d];
					//
					count = 0;
					suma2 = suma = 0;
					sumb2 = sumb = 0;
					sumab = 0;
					//
					for (k = 0; k < ninv_z; k++) {
						for (j = 0; j < ninv_y; j++) {
							for (i = 0; i < ninv_x; i++) {
								x1 = x10 + i - dx;
								y1 = y10 + j - dy;
								z1 = z10 + k - dz;
								if ((x1 <= 0) || (x1 >= vd1_x-1) || (y1 <= 0) || (y1 >= vd1_y-1) || (z1 <= 0) || (z1 >= vd1_z-1)) {
									//vd1 = -1;
									vd1 = 0;
								} else {
									ix1 = (int)x1;
									iy1 = (int)y1;
									iz1 = (int)z1;
									fx = x1 - ix1;
									fy = y1 - iy1;
									fz = z1 - iz1;
									if (fx == 0 && fy == 0 && fz == 0) {
										vd1 = vdata1[iz1  ][iy1  ][ix1  ][0];
									} else {
										fx1 = 1.0f - fx;
										fy1 = 1.0f - fy;
										fz1 = 1.0f - fz;
										vd1  = fx1*fy1*fz1*vdata1[iz1  ][iy1  ][ix1  ][0];
										vd1 += fx *fy1*fz1*vdata1[iz1  ][iy1  ][ix1+1][0];
										vd1 += fx1*fy *fz1*vdata1[iz1  ][iy1+1][ix1  ][0];
										vd1 += fx1*fy1*fz *vdata1[iz1+1][iy1  ][ix1  ][0];
										vd1 += fx *fy *fz1*vdata1[iz1  ][iy1+1][ix1+1][0];
										vd1 += fx *fy1*fz *vdata1[iz1+1][iy1  ][ix1+1][0];
										vd1 += fx1*fy *fz *vdata1[iz1+1][iy1+1][ix1  ][0];
										vd1 += fx *fy *fz *vdata1[iz1+1][iy1+1][ix1+1][0];
									}
								}
								//
								//if (vd1 < 0) {
								//	continue;
								//}
								//
								x2 = x20 + i + dx;
								y2 = y20 + j + dy;
								z2 = z20 + k + dz;
								if ((x2 <= 0) || (x2 >= vd2_x-1) || (y2 <= 0) || (y2 >= vd2_y-1) || (z2 <= 0) || (z2 >= vd2_z-1)) {
									vd2 = 0;
								} else {
									ix2 = (int)x2;
									iy2 = (int)y2;
									iz2 = (int)z2;
									fx = x2 - ix2;
									fy = y2 - iy2;
									fz = z2 - iz2;
									if (fx == 0 && fy == 0 && fz == 0) {
										vd2 = vdata2[iz2  ][iy2  ][ix2  ][0];
									} else {
										fx1 = 1.0f - fx;
										fy1 = 1.0f - fy;
										fz1 = 1.0f - fz;
										vd2  = fx1*fy1*fz1*vdata2[iz2  ][iy2  ][ix2  ][0];
										vd2 += fx *fy1*fz1*vdata2[iz2  ][iy2  ][ix2+1][0];
										vd2 += fx1*fy *fz1*vdata2[iz2  ][iy2+1][ix2  ][0];
										vd2 += fx1*fy1*fz *vdata2[iz2+1][iy2  ][ix2  ][0];
										vd2 += fx *fy *fz1*vdata2[iz2  ][iy2+1][ix2+1][0];
										vd2 += fx *fy1*fz *vdata2[iz2+1][iy2  ][ix2+1][0];
										vd2 += fx1*fy *fz *vdata2[iz2+1][iy2+1][ix2  ][0];
										vd2 += fx *fy *fz *vdata2[iz2+1][iy2+1][ix2+1][0];
									}
								}
								//
								suma  += vd1;
								suma2 += vd1 * vd1;
								sumb  += vd2;
								sumb2 += vd2 * vd2;
								sumab += vd1 * vd2;
								count += 1;
							}
						}
					}
					//
					if (count > 0) {
						fixedMean  = suma / count;
						movingMean = sumb / count;
						sff = suma2 - fixedMean*suma  - fixedMean*suma  + count*fixedMean*fixedMean;
						smm = sumb2 - movingMean*sumb - movingMean*sumb + count*movingMean*movingMean;
						sfm = sumab - movingMean*suma - fixedMean*sumb  + count*movingMean*fixedMean;
						//
						//if ((sff > 0) && (smm > 0)) {
						if ((sff > 1e-1) && (smm > 1e-1)) {
#ifdef USE_CC_NCC
							corr = sfm / sqrt(sff * smm);
#else
							corr = (sfm * sfm) / (sff * smm);
#endif

							if (corr > max_corr) {
								max_corr = corr;
							}
							if (corr < min_corr) {
								min_corr = corr;
							}
							//*
							if (corr > corr_max) {
								corr = corr_max;
							}
							//*/
						} else {
							corr = 0;
						}
					} else {
						corr = 0;
					}
					//
					dcv[n][m][l][d] = (float)(255 * 128 * (corr_max - corr));
				}
			}
		}
	}

	TRACE("\nmin_corr: %f, max_corr = %f\n", min_corr, max_corr);
}

template <class T>
void ComputeDataCost3DSyD_CC_Fast(T**** vdata1, int vd1_x, int vd1_y, int vd1_z, int vd1_s, T**** vdata2, int vd2_x, int vd2_y, int vd2_z, int vd2_s,
	REALV**** XC, REALV**** YC, REALV**** ZC,
	int mesh_x, int mesh_y, int mesh_z, int mesh_ex, int mesh_ey, int mesh_ez,
	REALV* disp_x, REALV* disp_y, REALV* disp_z, int num_d, REALV**** dcv,
	int dc_skip_back, T dc_back_color, int radius = 4) 
{
	int ninv_x, ninv_y, ninv_z;
	int ninv_cx, ninv_cy, ninv_cz;
	int ninv_size;
	int i, j, k, l, m, n, d;
	double max_corr, min_corr;
	REALV disp_x_max, disp_x_min, disp_y_max, disp_y_min, disp_z_max, disp_z_min;
	int lx, ly, lz, rx, ry, rz;
	DVolume vd1t, vd2t;
	int vdt_x, vdt_y, vdt_z;

	ninv_x = 2 * radius + 1;
	ninv_y = 2 * radius + 1;
	ninv_z = 2 * radius + 1;
	ninv_size = ninv_x * ninv_y * ninv_z;
	//
	ninv_cx = ninv_x / 2;
	ninv_cy = ninv_y / 2;
	ninv_cz = ninv_z / 2;

	disp_x_max = -10000; disp_x_min = 10000;
	disp_y_max = -10000; disp_y_min = 10000;
	disp_z_max = -10000; disp_z_min = 10000;
	for (d = 0; d < num_d; d++) {
		if (disp_x[d] > disp_x_max) { disp_x_max = disp_x[d]; }
		if (disp_x[d] < disp_x_min) { disp_x_min = disp_x[d]; }
		if (disp_y[d] > disp_y_max) { disp_y_max = disp_x[d]; }
		if (disp_y[d] < disp_y_min) { disp_y_min = disp_x[d]; }
		if (disp_z[d] > disp_z_max) { disp_z_max = disp_x[d]; }
		if (disp_z[d] < disp_z_min) { disp_z_min = disp_x[d]; }
	}
	
	lx = (int)(fabs(disp_x_min) + 0.999999) + ninv_cx;
	rx = (int)(fabs(disp_x_max) + 0.999999) + ninv_cx+1;
	ly = (int)(fabs(disp_y_min) + 0.999999) + ninv_cy;
	ry = (int)(fabs(disp_y_max) + 0.999999) + ninv_cy+1;
	lz = (int)(fabs(disp_z_min) + 0.999999) + ninv_cz;
	rz = (int)(fabs(disp_z_max) + 0.999999) + ninv_cz+1;

	vdt_x = max(vd1_x+lx+rx, vd2_x+lx+rx);
	vdt_y = max(vd1_y+ly+ry, vd2_y+ly+ry);
	vdt_z = max(vd1_z+lz+rz, vd2_z+lz+rz);

	vd1t.allocate(vdt_x, vdt_y, vdt_z);
	vd2t.allocate(vdt_x, vdt_y, vdt_z);
	
	max_corr = -10000;
	min_corr = 10000;
	for (d = 0; d < num_d; d++) {
		float xc, yc, zc;
		float x1, y1, z1;
		float x2, y2, z2;
		float dx, dy, dz;
		int ix1, iy1, iz1;
		int ix2, iy2, iz2;
		int ixc, iyc, izc;
		int ix, iy, iz;
		double vd1, vd2;
		double fx, fy, fz, fx1, fy1, fz1;
		double fixedMean, movingMean;
		double suma2, sumb2, suma, sumb, sumab, count;
		double sff, smm, sfm;
		double corr;
		double corr_max = 1.0;

		if (d % (int)(num_d * 0.1) == 0) {
			TRACE("processing %d / %d total d\n", d, num_d);
		}

		dx = disp_x[d];
		dy = disp_y[d];
		dz = disp_z[d];

		// translate images
		for (k = 0; k < vdt_z; k++) {
			for (j = 0; j < vdt_y; j++) {
				for (i = 0; i < vdt_x; i++) {
					x1 = i - (lx + dx);
					y1 = j - (ly + dy);
					z1 = k - (lz + dz);
					ix1 = (int)x1;
					iy1 = (int)y1;
					iz1 = (int)z1;
					fx = x1 - ix1;
					fy = y1 - iy1;
					fz = z1 - iz1;
					if (fx == 0 && fy == 0 && fz == 0) {
						if ((ix1 < 0) || (ix1 > vd1_x-1) || (iy1 < 0) || (iy1 > vd1_y-1) || (iz1 < 0) || (iz1 > vd1_z-1)) {
							vd1t.m_pData[k][j][i][0] = 0;
						} else {
							vd1t.m_pData[k][j][i][0] = vdata1[iz1][iy1][ix1][0];
						}
					} else {
						if ((ix1 < 0) || (ix1 >= vd1_x-1) || (iy1 < 0) || (iy1 >= vd1_y-1) || (iz1 < 0) || (iz1 >= vd1_z-1)) {
							vd1t.m_pData[k][j][i][0] = 0;
						} else {
							fx1 = 1.0f - fx;
							fy1 = 1.0f - fy;
							fz1 = 1.0f - fz;
							vd1  = fx1*fy1*fz1*vdata1[iz1  ][iy1  ][ix1  ][0];
							vd1 += fx *fy1*fz1*vdata1[iz1  ][iy1  ][ix1+1][0];
							vd1 += fx1*fy *fz1*vdata1[iz1  ][iy1+1][ix1  ][0];
							vd1 += fx1*fy1*fz *vdata1[iz1+1][iy1  ][ix1  ][0];
							vd1 += fx *fy *fz1*vdata1[iz1  ][iy1+1][ix1+1][0];
							vd1 += fx *fy1*fz *vdata1[iz1+1][iy1  ][ix1+1][0];
							vd1 += fx1*fy *fz *vdata1[iz1+1][iy1+1][ix1  ][0];
							vd1 += fx *fy *fz *vdata1[iz1+1][iy1+1][ix1+1][0];
							vd1t.m_pData[k][j][i][0] = vd1;
						}
					}

					x2 = i - (lx - dx);
					y2 = j - (ly - dy);
					z2 = k - (lz - dz);
					ix2 = (int)x2;
					iy2 = (int)y2;
					iz2 = (int)z2;
					fx = x2 - ix2;
					fy = y2 - iy2;
					fz = z2 - iz2;
					if (fx == 0 && fy == 0 && fz == 0) {
						if ((ix2 < 0) || (ix2 > vd2_x-1) || (iy2 < 0) || (iy2 > vd2_y-1) || (iz2 < 0) || (iz2 > vd2_z-1)) {
							vd2t.m_pData[k][j][i][0] = 0;
						} else {
							vd2t.m_pData[k][j][i][0] = vdata2[iz2][iy2][ix2][0];
						}
					} else {
						if ((ix2 < 0) || (ix2 >= vd2_x-1) || (iy2 < 0) || (iy2 >= vd2_y-1) || (iz2 < 0) || (iz2 >= vd2_z-1)) {
							vd2t.m_pData[k][j][i][0] = 0;
						} else {
							fx1 = 1.0f - fx;
							fy1 = 1.0f - fy;
							fz1 = 1.0f - fz;
							vd2  = fx1*fy1*fz1*vdata2[iz2  ][iy2  ][ix2  ][0];
							vd2 += fx *fy1*fz1*vdata2[iz2  ][iy2  ][ix2+1][0];
							vd2 += fx1*fy *fz1*vdata2[iz2  ][iy2+1][ix2  ][0];
							vd2 += fx1*fy1*fz *vdata2[iz2+1][iy2  ][ix2  ][0];
							vd2 += fx *fy *fz1*vdata2[iz2  ][iy2+1][ix2+1][0];
							vd2 += fx *fy1*fz *vdata2[iz2+1][iy2  ][ix2+1][0];
							vd2 += fx1*fy *fz *vdata2[iz2+1][iy2+1][ix2  ][0];
							vd2 += fx *fy *fz *vdata2[iz2+1][iy2+1][ix2+1][0];
							vd2t.m_pData[k][j][i][0] = vd2;
						}
					}
				}
			}
		}
		//vd1t.save("vd1t.nii.gz", 1);
		//vd2t.save("vd2t.nii.gz", 1);

		for (n = 0; n < mesh_z; n++) {
			for (m = 0; m < mesh_y; m++) {
				for (l = 0; l < mesh_x; l++) {
					xc = XC[n][m][l][0];
					yc = YC[n][m][l][0];
					zc = ZC[n][m][l][0];
					//
					if (dc_skip_back == 1) {
						if ((xc < 0) || (xc > vd1_x-1) || (yc < 0) || (yc > vd1_y-1) || (zc < 0) || (zc > vd1_z-1)) {
							dcv[n][m][l][d] = 0;
							continue;
						} else {
							//*
							if (vdata1[(int)zc][(int)yc][(int)xc][0] <= dc_back_color) {
								dcv[n][m][l][d] = 0;
								continue;
							}
							//*/
						}
					}
					//
					ixc = (int)(xc - ninv_cx + lx);
					iyc = (int)(yc - ninv_cy + ly);
					izc = (int)(zc - ninv_cz + lz);
					//
					//count = 0;
					suma2 = suma = 0;
					sumb2 = sumb = 0;
					sumab = 0;
					for (k = 0; k < ninv_z; k++) {
						for (j = 0; j < ninv_y; j++) {
							for (i = 0; i < ninv_x; i++) {
								ix = ixc + i;
								iy = iyc + j;
								iz = izc + k;
								//
								vd1 = vd1t.m_pData[iz][iy][ix][0];
								vd2 = vd2t.m_pData[iz][iy][ix][0];
								//
								suma  += vd1;
								suma2 += vd1 * vd1;
								sumb  += vd2;
								sumb2 += vd2 * vd2;
								sumab += vd1 * vd2;
								//count += 1;
							}
						}
					}
					count = ninv_size;
					//
					if (count > 0) {
						fixedMean  = suma / count;
						movingMean = sumb / count;
						sff = suma2 -  fixedMean*suma -  fixedMean*suma + count* fixedMean* fixedMean;
						smm = sumb2 - movingMean*sumb - movingMean*sumb + count*movingMean*movingMean;
						sfm = sumab - movingMean*suma -  fixedMean*sumb + count*movingMean* fixedMean;
						//
						//if ((sff > 0) && (smm > 0)) {
						//if ((sff > 1e-1) && (smm > 1e-1)) {
						if (sff*smm > 1.e-5) {
#ifdef USE_CC_NCC
							corr = sfm / sqrt(sff * smm);
#else
							corr = (sfm * sfm) / (sff * smm);
#endif

							if (corr > max_corr) {
								max_corr = corr;
							}
							if (corr < min_corr) {
								min_corr = corr;
							}
							//*
							if (corr > corr_max) {
								corr = corr_max;
							}
							//*/
						} else {
							corr = 0;
						}
					} else {
						corr = 0;
					}
					//
					dcv[n][m][l][d] = (float)(255 * 128 * (corr_max - corr));
				}
			}
		}
	}

	TRACE("\nmin_corr: %f, max_corr = %f\n", min_corr, max_corr);
}

template <class T>
void ComputeDataCost3D_NMI(T**** vdata1, int vd1_x, int vd1_y, int vd1_z, int vd1_s, T**** vdata2, int vd2_x, int vd2_y, int vd2_z, int vd2_s,
	REALV**** X1, REALV**** Y1, REALV**** Z1, REALV**** dX1, REALV**** dY1, REALV**** dZ1, REALV**** X2, REALV**** Y2, REALV**** Z2, REALV**** dX2, REALV**** dY2, REALV**** dZ2,
	int mesh_x, int mesh_y, int mesh_z, int mesh_ex, int mesh_ey, int mesh_ez,
	REALV* disp_x, REALV* disp_y, REALV* disp_z, int num_d, REALV**** dcv,
	int dc_weight, int dc_skip_back, T dc_back_color, int ninv_s = 1) 
{
	DC_REAL*** ninv;
	int ninv_x, ninv_y, ninv_z;
	int ninv_cx, ninv_cy, ninv_cz;
	DC_REAL ninv_size;
	int i, j, k, l, m, n, d;
	size_t nodeNum = mesh_x * mesh_y * mesh_z;
	DC_REAL corr;
	DC_REAL corr_min = 10;
	DC_REAL corr_max = -10;
	DC_REAL corr_mean = 0;
	int*** hist_1_patch;
	DC_REAL* hist_1;
	DC_REAL* hist_2;
	DC_REAL** hist_j;
	//
	int vd1_xm = vd1_x;
	int vd1_ym = vd1_y;
	int vd1_zm = vd1_z;
	int vd2_xm = vd2_x;
	int vd2_ym = vd2_y;
	int vd2_zm = vd2_z;
	int vd1_xo = 0;
	int vd1_yo = 0;
	int vd1_zo = 0;
	int vd2_xo = 0;
	int vd2_yo = 0;
	int vd2_zo = 0;
	//
	int vd1_x_1 = vd1_x - 1;
	int vd1_y_1 = vd1_y - 1;
	int vd1_z_1 = vd1_z - 1;
	int vd2_x_1 = vd2_x - 1;
	int vd2_y_1 = vd2_y - 1;
	int vd2_z_1 = vd2_z - 1;

	if (dc_weight == 0) {
		/*
		ninv_x = mesh_ex * 4 * ninv_s;
		ninv_y = mesh_ey * 4 * ninv_s;
		ninv_z = mesh_ez * 4 * ninv_s;
		/*/
		ninv_x = 2 * 4 * ninv_s;
		ninv_y = 2 * 4 * ninv_s;
		ninv_z = 2 * 4 * ninv_s;
		//*/
		ninv_cx = ninv_x / 2;
		ninv_cy = ninv_y / 2;
		ninv_cz = ninv_z / 2;
		//
		ninv_size = (float)(ninv_x * ninv_y * ninv_z);
		ninv = (DC_REAL***)malloc(ninv_z * sizeof(DC_REAL**));
		for (k = 0; k < ninv_z; k++) {
			ninv[k] = (DC_REAL**)malloc(ninv_y * sizeof(DC_REAL*));
			for (j = 0; j < ninv_y; j++) {
				ninv[k][j] = (DC_REAL*)malloc(ninv_x * sizeof(DC_REAL));
			}
		}
		get_ninv(ninv, ninv_x / 4, ninv_y / 4, ninv_z / 4);
	} else {
		/*
		ninv_x = mesh_ex * 2 * ninv_s;
		ninv_y = mesh_ey * 2 * ninv_s;
		ninv_z = mesh_ez * 2 * ninv_s;
		/*/
		ninv_x = 1 * 2 * ninv_s;
		ninv_y = 1 * 2 * ninv_s;
		ninv_z = 1 * 2 * ninv_s;
		//*/
		ninv_cx = ninv_x / 2;
		ninv_cy = ninv_y / 2;
		ninv_cz = ninv_z / 2;
		//
		ninv_size = (float)(ninv_x * ninv_y * ninv_z);
		DC_REAL _ninv_size = 1.0f / ninv_size;
		ninv = (DC_REAL***)malloc(ninv_z * sizeof(DC_REAL**));
		for (k = 0; k < ninv_z; k++) {
			ninv[k] = (DC_REAL**)malloc(ninv_y * sizeof(DC_REAL*));
			for (j = 0; j < ninv_y; j++) {
				ninv[k][j] = (DC_REAL*)malloc(ninv_x * sizeof(DC_REAL));
				for (i = 0; i < ninv_x; i++) {
					ninv[k][j][i] = _ninv_size;
				}
			}
		}
	}
	/*{
		FILE* fp;
		fp = fopen("ninv.txt", "w");
		for (k = 0; k < ninv_z; k++) {
			for (j = 0; j < ninv_y; j++) {
				for (i = 0; i < ninv_x; i++) {
					fprintf(fp, "%e ", ninv[k][j][i]);
				}
				fprintf(fp, "\n");
			}
		}
		fclose(fp);
	}*/
	{
		DC_REAL ninv_sum = 0;
		for (k = 0; k < ninv_z; k++) {
			for (j = 0; j < ninv_y; j++) {
				for (i = 0; i < ninv_x; i++) {
					ninv_sum += ninv[k][j][i];
				}
			}
		}
		TRACE("ninv_sum = %f\n", ninv_sum);
	}

	hist_1 = (DC_REAL*)malloc(NMI_COLOR_NUM * sizeof(DC_REAL));
	hist_2 = (DC_REAL*)malloc(NMI_COLOR_NUM * sizeof(DC_REAL));
	hist_j = (DC_REAL**)malloc(NMI_COLOR_NUM * sizeof(DC_REAL*));
	for (j = 0; j < NMI_COLOR_NUM; j++) {
		hist_j[j] = (DC_REAL*)malloc(NMI_COLOR_NUM * sizeof(DC_REAL));
	}
	//
	hist_1_patch = (int***)malloc(ninv_z * sizeof(int**));
	for (k = 0; k < ninv_z; k++) {
		hist_1_patch[k] = (int**)malloc(ninv_y * sizeof(int*));
		for (j = 0; j < ninv_y; j++) {
			hist_1_patch[k][j] = (int*)malloc(ninv_x * sizeof(int));
		}
	}

	for (n = 0; n < mesh_z; n++) {
		TRACE("processing %d / %d total z\n", n, mesh_z);
		for (m = 0; m < mesh_y; m++) {
			for (l = 0; l < mesh_x; l++) {
				DC_REAL x20, y20, z20;
				DC_REAL x2, y2, z2;
				DC_REAL x10, y10, z10;
				DC_REAL x1, y1, z1;
				DC_REAL dx, dy, dz;
				DC_REAL vd1, vd2;
				int ix1, iy1, iz1;
				int ix2, iy2, iz2;
				DC_REAL fx, fy, fz, fx1, fy1, fz1;
				//
				DC_REAL nw;
				//DC_REAL Hj, H1, H2, nmi;
				DC_REAL Hj_t, H1_t, H2_t, nmi_t;
				int v1, v2;
				//
				x10 = X1[n][m][l][0] + dX1[n][m][l][0] + vd1_xo - ninv_cx;
				y10 = Y1[n][m][l][0] + dY1[n][m][l][0] + vd1_yo - ninv_cy;
				z10 = Z1[n][m][l][0] + dZ1[n][m][l][0] + vd1_zo - ninv_cz;
				//
				if (dc_skip_back == 1) {
					x1 = x10 + ninv_cx;
					y1 = y10 + ninv_cy;
					z1 = z10 + ninv_cz;
					if ((x1 <= 0) || (x1 >= vd1_x-1) || (y1 <= 0) || (y1 >= vd1_y-1) || (z1 <= 0) || (z1 >= vd1_z-1)) {
						for (d = 0; d < num_d; d++) {
							dcv[n][m][l][d] = 0;
						}
						continue;
					} else {
						if (vdata1[(int)z1][(int)y1][(int)x1][0] <= dc_back_color) {
							for (d = 0; d < num_d; d++) {
								dcv[n][m][l][d] = 0;
							}
							continue;
						}
					}
				}
				//
				for (j = 0; j < NMI_COLOR_NUM; j++) {
					hist_1[j] = 0;
				}
				//
				for (k = 0; k < ninv_z; k++) {
					for (j = 0; j < ninv_y; j++) {
						for (i = 0; i < ninv_x; i++) {
							x1 = x10 + i;
							y1 = y10 + j;
							z1 = z10 + k;
							if ((x1 <= 0) || (x1 >= vd1_x_1) || (y1 <= 0) || (y1 >= vd1_y_1) || (z1 <= 0) || (z1 >= vd1_z_1)) {
								vd1 = 0;
							} else {
								ix1 = (int)x1;
								iy1 = (int)y1;
								iz1 = (int)z1;
								fx = x1 - ix1;
								fy = y1 - iy1;
								fz = z1 - iz1;
								if (fx == 0 && fy == 0 && fz == 0) {
									vd1 = vdata1[iz1  ][iy1  ][ix1  ][0];
								} else {
									fx1 = 1.0f - fx;
									fy1 = 1.0f - fy;
									fz1 = 1.0f - fz;
									vd1  = fx1*fy1*fz1*vdata1[iz1  ][iy1  ][ix1  ][0];
									vd1 += fx *fy1*fz1*vdata1[iz1  ][iy1  ][ix1+1][0];
									vd1 += fx1*fy *fz1*vdata1[iz1  ][iy1+1][ix1  ][0];
									vd1 += fx1*fy1*fz *vdata1[iz1+1][iy1  ][ix1  ][0];
									vd1 += fx *fy *fz1*vdata1[iz1  ][iy1+1][ix1+1][0];
									vd1 += fx *fy1*fz *vdata1[iz1+1][iy1  ][ix1+1][0];
									vd1 += fx1*fy *fz *vdata1[iz1+1][iy1+1][ix1  ][0];
									vd1 += fx *fy *fz *vdata1[iz1+1][iy1+1][ix1+1][0];
								}
							}
							//
							v1 = (int)vd1 >> NMI_COLOR_SHIFT;
							hist_1_patch[k][j][i] = v1;
							hist_1[v1] += ninv[k][j][i];
						}
					}
				}
				//
				//H1 = 0.0;
				H1_t = 0.0;
				for (j = 0; j < NMI_COLOR_NUM; j++) {
					//if (hist_1[j] != 0) {
					//	H1 += -(hist_1[j] * log(hist_1[j]));
					//}
					H1_t += (float)ENT(hist_1[j]);
				}
				//
				x20 = X2[n][m][l][0] + dX2[n][m][l][0] + vd2_xo - ninv_cx;
				y20 = Y2[n][m][l][0] + dY2[n][m][l][0] + vd2_yo - ninv_cy;
				z20 = Z2[n][m][l][0] + dZ2[n][m][l][0] + vd2_zo - ninv_cz;
				//
				for (d = 0; d < num_d; d++) {
					for (j = 0; j < NMI_COLOR_NUM; j++) {
						for (i = 0; i < NMI_COLOR_NUM; i++) {
							hist_j[j][i] = 0;
						}
						hist_2[j] = 0;
					}
					//
					dx = disp_x[d];
					dy = disp_y[d];
					dz = disp_z[d];
					//
					for (k = 0; k < ninv_z; k++) {
						for (j = 0; j < ninv_y; j++) {
							for (i = 0; i < ninv_x; i++) {
								x2 = x20 + i + dx;
								y2 = y20 + j + dy;
								z2 = z20 + k + dz;
								if ((x2 <= 0) || (x2 >= vd2_x_1) || (y2 <= 0) || (y2 >= vd2_y_1) || (z2 <= 0) || (z2 >= vd2_z_1)) {
									vd2 = 0;
								} else {
									ix2 = (int)x2;
									iy2 = (int)y2;
									iz2 = (int)z2;
									fx = x2 - ix2;
									fy = y2 - iy2;
									fz = z2 - iz2;
									if (fx == 0 && fy == 0 && fz == 0) {
										vd2 = vdata2[iz2  ][iy2  ][ix2  ][0];
									} else {
										fx1 = 1.0f - fx;
										fy1 = 1.0f - fy;
										fz1 = 1.0f - fz;
										vd2  = fx1*fy1*fz1*vdata2[iz2  ][iy2  ][ix2  ][0];
										vd2 += fx *fy1*fz1*vdata2[iz2  ][iy2  ][ix2+1][0];
										vd2 += fx1*fy *fz1*vdata2[iz2  ][iy2+1][ix2  ][0];
										vd2 += fx1*fy1*fz *vdata2[iz2+1][iy2  ][ix2  ][0];
										vd2 += fx *fy *fz1*vdata2[iz2  ][iy2+1][ix2+1][0];
										vd2 += fx *fy1*fz *vdata2[iz2+1][iy2  ][ix2+1][0];
										vd2 += fx1*fy *fz *vdata2[iz2+1][iy2+1][ix2  ][0];
										vd2 += fx *fy *fz *vdata2[iz2+1][iy2+1][ix2+1][0];
									}
								}
								//
								v1 = hist_1_patch[k][j][i];
								v2 = ((int)vd2) >> NMI_COLOR_SHIFT;
								nw = ninv[k][j][i];

								hist_j[v2][v1] += nw;
								hist_2[v2]     += nw;
							}
						}
					}
					//
					//Hj = H2 = 0.0;
					Hj_t = H2_t = 0.0;
					for (j = 0; j < NMI_COLOR_NUM; j++) {
						for (i = 0; i < NMI_COLOR_NUM; i++) {
							//if (hist_j[j][i] != 0) {
							//	Hj += -(hist_j[j][i] * log(hist_j[j][i]));
							//}
							Hj_t += (float)ENT(hist_j[j][i]);
						}
						//if (hist_2[j] != 0) {
						//	H2 += -(hist_2[j] * log(hist_2[j]));
						//}
						H2_t += (float)ENT(hist_2[j]);
					}

					if (Hj_t == 0) {
						nmi_t = 1;
					} else {
						//nmi = (H1 + H2) / Hj;
						nmi_t = (H1_t + H2_t) / Hj_t;
						//return -nmi;
						//return -nmi_t;
					}

					corr = -nmi_t;

					/*
					if (H1_t != 0 && H2_t != 0) {
						corr = corr;
					}

					if (corr_min > corr) {
						corr_min = corr;
					}
					if (corr_max < corr) {
						corr_max = corr;
					}
					//*/
					//
					//dcv[n][m][l][d] = 255 * 256 * (2 + corr);
					dcv[n][m][l][d] = 85 * 256 * (2 + corr);
				}
			}
		}
	}

	//TRACE("corr_min = %f, corr_max = %f\r\n", corr_min, corr_max);

	for (k = 0; k < ninv_z; k++) {
		for (j = 0; j < ninv_y; j++) {
			free(ninv[k][j]);
		}
		free(ninv[k]);
	}
	free(ninv);
	//
	free(hist_1);
	free(hist_2);
	for (j = 0; j < NMI_COLOR_NUM; j++) {
		free(hist_j[j]);
	}
	free(hist_j);
	//
	for (k = 0; k < ninv_z; k++) {
		for (j = 0; j < ninv_y; j++) {
			free(hist_1_patch[k][j]);
		}
		free(hist_1_patch[k]);
	}
	free(hist_1_patch);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// image-wise score
template <class T>
BOOL ComputeQ_NMI(T**** vdata1, unsigned char**** mask1, T**** vdata2, unsigned char**** mask2, int vd_x, int vd_y, int vd_z, double* pScore, int tag = 0, char* tag_F = NULL) 
{
	int i, j, l, m, n;
	float corr = 0;
	double* hist_1;
	double* hist_2;
	double** hist_j;
	int hist_num = 0;
	int mask_num = 0;
	
	if (tag_F != NULL) {
		TRACE("%d_%s: ComputeQ_NMI...\n", tag, tag_F);
	} else {
		TRACE("ComputeQ_NMI...\n");
	}

	hist_1 = (double*)malloc(NMI_COLOR_NUM * sizeof(double));
	hist_2 = (double*)malloc(NMI_COLOR_NUM * sizeof(double));
	hist_j = (double**)malloc(NMI_COLOR_NUM * sizeof(double*));
	for (j = 0; j < NMI_COLOR_NUM; j++) {
		hist_j[j] = (double*)malloc(NMI_COLOR_NUM * sizeof(double));
	}

	for (j = 0; j < NMI_COLOR_NUM; j++) {
		for (i = 0; i < NMI_COLOR_NUM; i++) {
			hist_j[j][i] = 0;
		}
		hist_1[j] = 0;
		hist_2[j] = 0;
	}

	for (n = 0; n < vd_z; n++) {
		for (m = 0; m < vd_y; m++) {
			for (l = 0; l < vd_x; l++) {
				T vd1, vd2;
				int v1, v2;
				//
				vd1 = vdata1[n][m][l][0];
				vd2 = vdata2[n][m][l][0];
				//
				if (vd1 <= 0 || vd2 <= 0) {
					continue;
				}
				if ((vd1 >= 256) || (vd2 >= 256)) {
					continue;
				}
				//
				if (mask1[n][m][l][0] == 1) {
					mask_num++;
					continue;
				}
				if (mask2[n][m][l][0] == 1) {
					mask_num++;
					continue;
				}
				//
				v1 = (int)(vd1 * NMI_COLOR_MUL);
				v2 = (int)(vd2 * NMI_COLOR_MUL);
				//
				hist_1[v1]	   += 1;
				hist_j[v2][v1] += 1;
				hist_2[v2]     += 1;
				hist_num	   += 1;
			}
		}
	}

	if (hist_num <= 0) {
		goto errret;
	}

	for (j = 0; j < NMI_COLOR_NUM; j++) {
		for (i = 0; i < NMI_COLOR_NUM; i++) {
			hist_j[j][i] /= hist_num;
		}
		hist_1[j] /= hist_num;
		hist_2[j] /= hist_num;
	}

	{
		double Hj_t, H1_t, H2_t, nmi_t;

		Hj_t = H1_t = H2_t = 0.0;
		for (j = 0; j < NMI_COLOR_NUM; j++) {
			for (i = 0; i < NMI_COLOR_NUM; i++) {
				Hj_t += ENT(hist_j[j][i]);
			}
			H1_t += ENT(hist_1[j]);
			H2_t += ENT(hist_2[j]);
		}

		if (Hj_t == 0) {
			nmi_t = 1;
		} else {
			nmi_t = (H1_t + H2_t) / Hj_t;
		}

		corr = (float)(2.0 - nmi_t);
	}

errret:
	free(hist_1);
	free(hist_2);
	for (j = 0; j < NMI_COLOR_NUM; j++) {
		free(hist_j[j]);
	}
	free(hist_j);

	*pScore = corr;

	if (tag_F != NULL) {
		TRACE("%d_%s: corr = %f, hist_num = %d, mask_num = %d\n", tag, tag_F, corr, hist_num, mask_num);
		TRACE("%d_%s: ComputeQ_NMI... - done\n", tag, tag_F);
	} else {
		TRACE("corr = %f, hist_num = %d, mask_num = %d\n", corr, hist_num, mask_num);
		TRACE("ComputeQ_NMI... - done\n");
	}

	return TRUE;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
