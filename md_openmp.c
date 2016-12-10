/*
	This code is using Molecular dynamics to study the evolution of the electron bunch form the photo cathod in Ultrafast electron diffraction experiment.

	origian Fortran code version: MD_1121.f90

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <fftw3.h>

//simulation setup
#define N 5000					//number of particles
#define Ntime 1 
#define newR 2.0E6		//size of the simulation box ~ 100 um
#define cutoff 200.0	//lower limit of the initial distance between electrons ~ 100 nm

//PPPM setup
#define bn 10					//number of boxes per direction
#define boxcap 50		//temporary cap for particles in one box; update to class later
#define hx 2.0E5			//[real space] cell size newR/bn
#define hx3 8.0E15		//hx^3 as the cell volume
#define grid_offset 5	//move the center of grid to 0 to match the particles
#define bn3 1000			// total grid/mesh point number

//parameters
#define m 5.4858e-4		//elelctron mass
#define vc 5.85e3			//speed of light
#define PI 3.141592653589793	//  \pi

double getrand(){ return (double)rand()/(double)RAND_MAX; }

/*	
double PBC(double r1) {
	double pr;
	if (r1 > 0.0) {
	  r1 -= floor(r1/newR)*newR;
	  pr = (r1 < (0.5*newR)) ? r1 : (r1 - newR) ;
	}
	else if (r1 < 0.0) {
	  r1 += ceil(r1/newR)*newR;
	  pr = (r1 > (-0.5*newR))? r1 : (r1 + newR) ;
	}
	else {
		pr = r1;
	}
	if (abs(pr)> 0.5*newR) {printf("pbc failed! \n");}
	return pr;
}
*/

double getPE(double r[N][3]){
	int i,j,k;
	double rel,vij;
	double PE_c = 0.0;
	for(j=0; j<(N-1); j++) {
		for(i=j+1; i<N; i++) {
			vij = 0.0;
			for(k=0; k<3; k++) {
				rel = r[i][k] - r[j][k];
				vij += pow(rel,2.0); //vij = r^2 for now
			}
			PE_c += pow(vij,-0.5); // r^(-1)
		}
	}
	return PE_c;
}

double getKE(double v[N][3]){
	int i,j;
	double KE_c = 0.0; //sum(v**2)
	for (i=0; i<N; i++){
		for (j=0; j<3; j++) {
			KE_c += pow(v[i][j],2.0);
		}
	}
	KE_c = 0.5*m*KE_c;
	return KE_c;
}

int pbc_box(int point){
	int pb;
	if (point < 0 ) {
		pb = point + bn;
	}
	else if (point > (bn-1)) {
		pb = point - bn;
	}
	else {
		pb = point;
	}
	return pb;
}

void charge_assign(int b[3], double basis[3],double rho[bn][bn][bn],int id,double w[N][2][2][2]){
	int dx,dy,dz,pbx,pby,pbz;
	double lx,ly,lz,weight;
	for (dx=0; dx<2; dx++){
		pbx = pbc_box(b[0]+dx);
	  lx = (dx == 0) ? (hx - basis[0]) : basis[0];
	  for (dy=0; dy<2; dy++){
	    pby = pbc_box(b[1]+dy);
	    ly = (dy == 0) ? (hx - basis[1]) : basis[1];
	    for (dz=0; dz<2; dz++){
	      pbz = pbc_box(b[2]+dz);
	      lz = (dz == 0) ? (hx - basis[2]) : basis[2];
	      weight = lx*ly*lz/hx3;
	      w[id][pbx][pby][pbz] = weight;
	      rho[pbx][pby][pbz] +=  weight;
	    }
	  }
	}
}

// assign electron into bn*bn*bn boxes and store their idx
void update_box(double r[N][3], int box[bn][bn][bn][boxcap],int boxid[N][3],double rho[bn][bn][bn],double w[N][2][2][2]) {
	int i,j,k,realb[3],b[3],num;
	double basis[3];
	for (i=0; i<bn; i++){
		for (j=0; j<bn; j++){
			for (k=0; k<bn; k++){
				box[i][j][k][0]=0;
				rho[i][j][k] = 0.0;
			}
		}
	}
	printf("update box starts\n");
	for (i=0; i<N; i++) {
//		printf("i = %d \n",i);
		for (j=0; j<3; j++) {
			realb[j] = (int)floor(r[i][j]/hx);
			//b[j] = (int)floor(r[i][j]/hx) + grid_offset;
			b[j] = (int)floor(r[i][j]/hx);
//			printf("%d ",b[j]);
			boxid[i][j] = b[j];
//			printf("%d ",boxid[i][j]);
			basis[j] = r[i][j] - realb[j]*hx;
		}
//		printf("\n");
		num = box[b[0]][b[1]][b[2]][0] + 1;
		box[b[0]][b[1]][b[2]][0] = num;
		box[b[0]][b[1]][b[2]][num] = i;
		charge_assign(b,basis,rho,i,w);	
	}
}

void setupKpoints(double kpoints[bn]){
	double k;
	int i;
	double nyquist = (double)bn/newR/2.0;
	for (i=0; i<(bn/2+1); i++){
		kpoints[i] = 2*PI*(double)i/(double)(bn/2)*nyquist;
		if (i!=(bn/2) && i!=0) {
			kpoints[bn-i] = kpoints[i];
		}
	}
}

void getGlobalField(double rho[bn][bn][bn],double field[3][bn][bn][bn]) {
	fftw_complex 	*f_fft_result, *b_fft_in_x, *b_fft_in_y, *b_fft_in_z;
	fftw_plan			plan_forward, plan_backward_x, plan_backward_y, plan_backward_z;
	double kpoints[bn];
	int i,j,k,idx;
	double fr,fi; // for real part and imaginary part 
	double kx,ky,kz,kx2,ky2,kz2,k2i;

	f_fft_result  = ( fftw_complex* ) fftw_malloc( sizeof( fftw_complex ) * bn3 );
	b_fft_in_x 		= ( fftw_complex* ) fftw_malloc( sizeof( fftw_complex ) * bn3 );
	b_fft_in_y 		= ( fftw_complex* ) fftw_malloc( sizeof( fftw_complex ) * bn3 );
	b_fft_in_z 		= ( fftw_complex* ) fftw_malloc( sizeof( fftw_complex ) * bn3 );

	plan_forward  	= fftw_plan_dft_r2c_3d(bn,bn,bn,&rho[0][0][0],f_fft_result,FFTW_ESTIMATE );
	plan_backward_x = fftw_plan_dft_c2r_3d(bn,bn,bn,b_fft_in_x,&field[0][0][0][0],FFTW_ESTIMATE );
	plan_backward_y = fftw_plan_dft_c2r_3d(bn,bn,bn,b_fft_in_y,&field[1][0][0][0],FFTW_ESTIMATE );
	plan_backward_z = fftw_plan_dft_c2r_3d(bn,bn,bn,b_fft_in_z,&field[2][0][0][0],FFTW_ESTIMATE );

	printf("setupPPPM\n");
/*	
	for(i=0; i<bn; i++) {
		fprintf(stdout, "rho[%d]=%11.3f \n", i, rho[0][0][i]);
	}
*/
	
	fftw_execute( plan_forward );
	printf("finish forward\n");

	setupKpoints(kpoints);//since newR =2E6, k might be too small
	printf("setupKpoints\n");
	for (i=0; i< bn; i++){
		kx =  kpoints[i];
		kx2 = kx*kx;
		for (j=0; j<bn; j++){
			ky = kpoints[j];
			ky2 = ky*ky;
			for (k=0; k<(bn/2+1); k++) {
				kz = kpoints[k];
				kz2 = kz*kz;
				idx = 2*(k+(bn/2+1)*j+bn*(bn/2+1)*i);
				if (i==0 && j==0 && k==0) {
					k2i = 1.0;
				}
				else {
					k2i = 1/(kx2+ky2+kz2);
				}
				idx = 2*(bn*(bn/2+1)*i + (bn/2+1)*j + k);
				fr = f_fft_result[idx][0];
				fi = f_fft_result[idx][1];
				
				b_fft_in_x[idx][0] = kx*k2i*fi;
				b_fft_in_x[idx][1] = -kx*k2i*fr;
				b_fft_in_y[idx][0] = ky*k2i*fi;
				b_fft_in_y[idx][1] = -ky*k2i*fr;
				b_fft_in_z[idx][0] = kz*k2i*fi;
				b_fft_in_z[idx][1] = -kz*k2i*fr;
			}
		}
	}
	printf("output==>input\n");
  fftw_execute( plan_backward_x );
  fftw_execute( plan_backward_y );
  fftw_execute( plan_backward_z );
/*
	for(i=0; i<bn; i++) {
		printf("result[%d]=%11.3f \n", i, field[0][0][0][i]);
	}
*/
//	printf("start free fftw\n");
//	//free memory
//	fftw_destroy_plan( plan_forward );
//	printf("finish free plan_forward\n");
//	fftw_destroy_plan( plan_backward_x );
//	fftw_destroy_plan( plan_backward_y );
//	fftw_destroy_plan( plan_backward_z );
//	
//	fftw_free( f_fft_result );
//	fftw_free( b_fft_in_x );
//	fftw_free( b_fft_in_y );
//	fftw_free( b_fft_in_z );
//	printf("finish free\n");
}


void getForce(double f[N][3],double r[N][3], int box[bn][bn][bn][boxcap],int boxid[N][3],double rho[bn][bn][bn],double w[N][2][2][2]) {
	int i,j,k,fx,b[3],dbx,dby,dbz,pbx,pby,pbz,nb_j;
	double rel[3], rel_c, rel_c2, fij[3], field[3][bn][bn][bn], ri[3], rj[3];
	double dummy;
	int jx,jy,jz,ijx,ijy,ijz;
	double jrel2;

	getGlobalField(rho,field);

//	#pragma omp parallel for private(bx,by,bz,pbx,pby,pbz,dbx,dby,dbz,nb_j,rel_c2,j,k,dummy)
	for (i=0; i<N; i++) {
		for (fx=0; fx<3; fx++){
			b[fx] = boxid[i][fx];
			ri[fx] = r[i][fx];
		}
		for (dbx=0; dbx<3; dbx++){
			pbx = pbc_box(b[0]+dbx-1);
			for (dby=0; dby<3; dby++){
				pby = pbc_box(b[1]+dby-1);
				for (dbz=0; dbz<3; dbz++){
					pbz = pbc_box(b[2]+dbz-1);
					for (j=1; j<(box[pbx][pby][pbz][0]+1); j++){
						nb_j = box[pbx][pby][pbz][j];
						if (nb_j != i) {
//*******PP*******
							rel_c2 = 0.0;
							for (k=0; k<3; k++) {
								rj[k] = r[nb_j][k];
								dummy = ri[k]-rj[k];
								rel[k] = dummy - floor(dummy/newR)*newR;
								rel_c2 += rel[k]*rel[k];
							}
							for(k=0; k<3; k++) {
							//	if (rel_c2 < 1e-6) {rel_c2 = 1e-6; printf("small number!");}
								f[i][k] += rel[k]/pow(rel_c2,1.5);
							}
//*******PM*******
							//cancel nb_j on neighboring grid point for PM
							for (jx=0; jx<2; jx++){
								for (jy=0; jy<2; jy++){
									for (jz=0; jz<2; jz++){
										//get to each neighboring gird point
										ijx = pbc_box(b[0]+jx);
										ijy = pbc_box(b[1]+jy);
										ijz = pbc_box(b[2]+jz);
										jrel2 = pow((rj[0]-ijx*hx),2.0)+pow((rj[1]-ijy*hx),2.0)+pow((rj[2]-ijz*hx),2.0);
										field[0][ijx][ijy][ijz] -= (ijx*hx-rj[0])/pow(jrel2,1.5);
										field[1][ijx][ijy][ijz] -= (ijy*hx-rj[1])/pow(jrel2,1.5);	
										field[2][ijx][ijy][ijz] -= (ijz*hx-rj[2])/pow(jrel2,1.5);	
									}
								}
							}
						}
					}
				}
			}
		}
		//PM
		for (dbx=0; dbx<2; dbx++){
			pbx = pbc_box(b[0]+dbx);
			for (dby=0; dby<2; dby++){
				pby = pbc_box(b[1]+dby);
				for (dbz=0; dbz<2; dbz++){
					pbz = pbc_box(b[2]+dbz);
					for (fx=0; fx<3; fx++){
						f[i][fx] += w[i][dbx][dby][dbz]*field[fx][dbx][dby][dbz];
					}
				}
			}
		}

	}
	printf("getforce finish\n");

/********** full interaction version	
	//use openmp here with (rel[3],rel_c,fij) private
	#pragma omp parallel for private(i,j,k,rel,rel_c,fij)
	for (j=0; j<(N-1); j++) {
		for (i=j+1; i<N; i++) {
			rel_c = 0.0;
			for (k=0; k<3; k++) {
				rel[k] = r[i][k] - r[j][k];
				rel[k] -= rint(rel[k]/newR)*newR;
				rel_c += pow(rel[k], 2.0 );
			}
			rel_c = pow(rel_c, -1.5 );
			for (k=0; k<3; k++) {
				fij[k] = rel[k]*rel_c;
				//atomic operation here
				#pragma omp atomic
				f[j][k] -= fij[k];
				#pragma omp atomic
				f[i][k] += fij[k];
			}
		}
	}
*********/
}

void verlet(double r[N][3],double v[N][3],double f[N][3], int box[bn][bn][bn][boxcap],int boxid[N][3],double rho[bn][bn][bn],double w[N][2][2][2], double dt){
	int i,j;
	double hdtm = 0.5*dt/m;
	for (i=0; i<N; i++) {
		for (j=0; j<3; j++) {
			v[i][j] += f[i][j]*hdtm;
			r[i][j] += v[i][j]*dt;
			r[i][j] = r[i][j]-floor(r[i][j]/newR)*newR;
			f[i][j] = 0.0; //setup for force calculation
		}
	}
	update_box(r,box,boxid,rho,w);
	getForce(f,r,box,boxid,rho,w);
	for (i=0; i<N; i++) {
		for (j=0; j<3; j++) {
			v[i][j] += f[i][j]*hdtm;
		}
	}
	printf("update v finish\n");
}

int main() {
	double R[N][3] = {{0.0}}, V[N][3] = {{0.0}}, F[N][3]={{0.0}};
	int box[bn][bn][bn][boxcap]; // need to go to class or dynamic array later 
	// [0] is for number of electrons in the box, and then [1]-[number] is the electron id
	int boxid[N][3];
	int i, iter;
	int numb, check;
	double dt = 2e-4, realt = 0.0;
	int plotstride = 20;
	double r0,r1,r2,rel0,rel1,rel2;
	double KE,PE;
	double rho[bn][bn][bn], w[N][2][2][2];
//	FILE *RVo,*To,*initR;
//
//	RVo = fopen("./RandV.dat","w+");
//	To = fopen("./time.dat","w+");
//	initR = fopen("./initR.xyz","w+");

	// Initialization in a sphere
	srand(time(NULL));
	numb = 0;
	while (numb < N) {
//		r0 = newR*(2*getrand()-1);
//		r1 = newR*(2*getrand()-1);
//		r2 = newR*(2*getrand()-1);
		r0 = newR*getrand();
		r1 = newR*getrand();
		r2 = newR*getrand();

		check = 0;
		if (sqrt(r0*r0+r1*r1+r2*r2) < newR) {
			for (i = 0; i < numb; i++){
				rel0 = R[i][0] - r0;
				rel1 = R[i][1] - r1;
				rel2 = R[i][2] - r2;
				if (sqrt(rel0*rel0+rel1*rel1+rel2*rel2) < cutoff) {
					check = 1;
					break;
				}				
			}
			if (check == 0) {
				R[numb][0] = r0;
				R[numb][1] = r1;
				R[numb][2] = r2;
				numb += 1;
			}
		}
	}
	
	//output initR for check
//	fprintf(initR,"%d \n", N);
//	fprintf(initR,"%d \n", 0);
//	for (i = 0; i<N; i++) {
//		fprintf(initR,"%5d %11.3f \t %11.3f \t %11.3f \n",1,R[i][0],R[i][1],R[i][2]);
//	}

	for (iter = 0; iter< Ntime; iter++) {
		printf("iter %d: \n", iter);
		verlet(R,V,F,box,boxid,rho,w,dt);
		realt += dt;
		printf("update realtime \n");
/*
//output
		if ((iter % plotstride) == 1) {		
			PE = getPE(R);
			KE = getKE(V);
//			printf("%11.5f \t %11.5f \t %11.5f \t %11.5f \n", realt, PE, KE, PE+KE);
			//position output
			fprintf(initR,"%d \n", N);
			fprintf(initR,"%d \n", iter);
			for (i = 0; i<N; i++) {
				fprintf(initR,"%5d %11.3f \t %11.3f \t %11.3f \n",1,R[i][0],R[i][1],R[i][2]);
			}
		}

		if (iter == 200) dt = 5.0;
		if (iter == 500) dt = 15.0;
		if (iter == 700) dt = 50.0;
		if (iter == 2000) plotstride = 100;
*/
	}


//	fclose(RVo);
//	fclose(To);
//	fclose(initR);

	return 0;

}
