//TODO: Figure out how small of a time-step we need to run
//for AB3 to not die
//AB2 survives at 128x128

//stuff is messing up near the boundaries and it looks like
//the problem originates in the z-field BC's
//going to try switching sign of those bc's.
//that didn't help, switching signs back and turning off NH
//terms all together.
//turning off NH terms did not stabilize it, turning back on.
//trying to run with half amplitude. -> lasted longer, but still messes up
//trying to turn off filter.

#include "TArray.hpp"
#include <blitz/array.h>
#include "T_util.hpp"
#include "Par_util.hpp"
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include "Splits.hpp"
#include <random/normal.h>
#include <fstream>
#include "Parformer.hpp"
#include <stdlib.h>
#include <stdio.h>
#include "gmres.hpp"
#include "gmres_2d_solver.hpp"
#include "gmres_2d_solver_impl.hpp"
#include "grad.hpp"
#include "chebmat.hpp"
#include "NSIntegrator.hpp"

using namespace TArrayn;
using namespace Transformer;

using blitz::Array;
using blitz::TinyVector;
using blitz::GeneralArrayStorage;
using blitz::Range;

using ranlib::Normal;

using namespace std;
using namespace NSIntegrator;
using namespace Timestep;

// Grid size

//#define Nx 1024
//#define Ny 128

#define Nx 64
#define Ny 64

//#define Nx 16384
//#define Ny 512

// #define Nx 128 //stuff blowing up near top boundary. why???
// #define Ny 128 //blows up for 128x128 and bigger

// Grid lengths

//#define Lx 2000.0
//#define Ly 4000.0 //using these works

#define Lx 60000.0   //using these does
#define Ly 10000.0

// Defines for physical parameters
#define G (0.01*9.81)
#define EARTH_OMEGA (2*M_PI/(24*3600))
#define EARTH_RADIUS (6371e3)
#define LATITUDE (M_PI/2)
//#define ROT_F (10.0*1.5e-3)
//#define ROT_F (7.8828e-5)
#define ROT_F (1.0e-4)
//#define ROT_F (2*EARTH_OMEGA*sin(LATITUDE))
#define ROT_B (0*2*EARTH_OMEGA*cos(LATITUDE)/EARTH_RADIUS)
#define H2_0 (5.0) 
#define H2_DEPTH (H2_0 + 0*cos(M_PI*y(kk)/Ly)*sin(4*M_PI*x(ii)/Lx))
#define H1_0 (15.0)
#define H1_DEPTH (H1_0 + 0*kk)
// Can probably include defines for initial conditions here.

// Timestep parameters
#define FINAL_T (3600.0*24.0)
//#define FINAL_T (3600.0*24.0*3.0)
#define INITIAL_T (0.0)
#define SAFETY_FACTOR (0.1) //changed from .25 as of Aug 3, 2013.
//#define SAFETY_FACTOR 0.0025
#define NUMOUTS (200.0)

// Filtering parameters
#define FILTER_ON true
#define F_CUTOFF (0.1)  //0.4 usually. 0.4 and 0.9 makes it die.
#define F_ORDER (4)
//#define F_STREN (-0.33)  // ask Chris why negative.
//#define F_STREN (1e-15)
//#define F_STREN (34.5) //this should give 1e-15 values at high wavenumber.
#define F_STREN (.1) //this should give 1e-8 values at high wavenumber.
//#define F_STREN (1e-2) // this smooths things out
//#define F_STREN (1e-15)  //this seems to do jack shit.

// GMRES parameters
#define MAXIT (40)
#define TOL (1.0e-11)
#define RESTARTS (1)
#define NOISYGMRES (false)

// do we want to output at all?
#define OUTFLAG true
#define NHOUTFLAG true //output auxiliary variable, z?

//time-stepper info
#define AB2 1
#define AB3 2
#define TIMESTEPPER (AB2) // can be AB2 or AB3. AB3 is unstable, not sure why

// Blitz index placeholders

blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

int main(int argc, char ** argv) {
   // Initialize MPI
   MPI_Init(&argc, &argv);

   // make grids
   Array<double,1> x(Nx), y(Ny);
   // matrices for dealing with imposing Neuman bc's on eta
   Array<double,2> Dy(Ny,Ny);
   Array<double,2> neumatinv(2,2);

   x = (ii+0.5)/Nx*2*M_PI;
   y = -cos(M_PI*ii/(Ny-1));

   // get cheb matrix on [-1,1]
   Dy = chebmat(Ny);

   //adjust from standard grids to physical grids
   x = (Lx/2/M_PI)*x(ii);
   y = Ly*((y(ii)+1)/2);

   //apply Jacobian to get cheb matrix on physical grid.
   Dy = (-2.0/Ly)*Dy; 
   // put entries into 2x2 Neuman-imposing matrix
   double neudet = Dy(0,0)*Dy(Ny-1,Ny-1)-Dy(0,Ny-1)*Dy(Ny-1,0);
   neumatinv(0,0) = Dy(Ny-1,Ny-1);
   neumatinv(0,1) = -Dy(0,Ny-1);
   neumatinv(1,0) = -Dy(Ny-1,0);
   neumatinv(1,1) = Dy(0,0);
   neumatinv = (1/neudet)*neumatinv;

   Stepped<double> times(4);
   //Stepped<double> coeffs_left(4);
   Stepped<double> coeffs_right(4);
   
   // Get parameters for local array storage
   TinyVector<int,3> local_lbound, local_extent;
   GeneralArrayStorage<3> local_storage;

   local_lbound = alloc_lbound(Nx,1,Ny);
   local_extent = alloc_extent(Nx,1,Ny);
   local_storage = alloc_storage(Nx,1,Ny);

   // Necessary FFT transformers (for filtering)
   TransWrapper XY_xform(Nx,1,Ny,FOURIER,NONE,CHEBY);

   // Allocate space for flow variables (2-timesteps for AB)
   vector<DTArray *> u1_levels(2);
   vector<DTArray *> u2_levels(2);
   vector<DTArray *> v1_levels(2);
   vector<DTArray *> v2_levels(2);  
   vector<DTArray *> eta_levels(2);
   vector<DTArray *> h1_levels(2);
   vector<DTArray *> h2_levels(2);
   vector<DTArray *> hu1_levels(2);
   vector<DTArray *> hu2_levels(2);
   vector<DTArray *> hv1_levels(2);
   vector<DTArray *> hv2_levels(2);

   vector<DTArray *> rhsh2_levels(3);
   vector<DTArray *> rhshu1_levels(3);
   vector<DTArray *> rhshu2_levels(3);
   vector<DTArray *> rhshv1_levels(3);
   vector<DTArray *> rhshv2_levels(3);
   
   //Allocate the arrays used in the above
   u1_levels[0] = new DTArray(local_lbound,local_extent,local_storage);
   u1_levels[1] = new DTArray(local_lbound,local_extent,local_storage);
   u2_levels[0] = new DTArray(local_lbound,local_extent,local_storage);
   u2_levels[1] = new DTArray(local_lbound,local_extent,local_storage);

   v1_levels[0] = new DTArray(local_lbound,local_extent,local_storage);
   v1_levels[1] = new DTArray(local_lbound,local_extent,local_storage);
   v2_levels[0] = new DTArray(local_lbound,local_extent,local_storage);
   v2_levels[1] = new DTArray(local_lbound,local_extent,local_storage);

   eta_levels[0] = new DTArray(local_lbound,local_extent,local_storage);
   eta_levels[1] = new DTArray(local_lbound,local_extent,local_storage);   

   h1_levels[0] = new DTArray(local_lbound,local_extent,local_storage);
   h1_levels[1] = new DTArray(local_lbound,local_extent,local_storage);
  
   h2_levels[0] = new DTArray(local_lbound,local_extent,local_storage);
   h2_levels[1] = new DTArray(local_lbound,local_extent,local_storage);

   hu1_levels[0] = new DTArray(local_lbound,local_extent,local_storage);
   hu1_levels[1] = new DTArray(local_lbound,local_extent,local_storage);
   hu2_levels[0] = new DTArray(local_lbound,local_extent,local_storage);
   hu2_levels[1] = new DTArray(local_lbound,local_extent,local_storage);

   hv1_levels[0] = new DTArray(local_lbound,local_extent,local_storage);
   hv1_levels[1] = new DTArray(local_lbound,local_extent,local_storage);
   hv2_levels[0] = new DTArray(local_lbound,local_extent,local_storage);
   hv2_levels[1] = new DTArray(local_lbound,local_extent,local_storage);

   rhsh2_levels[0] = new DTArray(local_lbound,local_extent,local_storage);
   rhsh2_levels[1] = new DTArray(local_lbound,local_extent,local_storage);
   rhsh2_levels[2] = new DTArray(local_lbound,local_extent,local_storage);

   rhshu1_levels[0] = new DTArray(local_lbound,local_extent,local_storage);
   rhshu1_levels[1] = new DTArray(local_lbound,local_extent,local_storage);
   rhshu1_levels[2] = new DTArray(local_lbound,local_extent,local_storage);

   rhshu2_levels[0] = new DTArray(local_lbound,local_extent,local_storage);
   rhshu2_levels[1] = new DTArray(local_lbound,local_extent,local_storage);
   rhshu2_levels[2] = new DTArray(local_lbound,local_extent,local_storage);

   rhshv1_levels[0] = new DTArray(local_lbound,local_extent,local_storage);
   rhshv1_levels[1] = new DTArray(local_lbound,local_extent,local_storage);
   rhshv1_levels[2] = new DTArray(local_lbound,local_extent,local_storage);

   rhshv2_levels[0] = new DTArray(local_lbound,local_extent,local_storage);
   rhshv2_levels[1] = new DTArray(local_lbound,local_extent,local_storage);
   rhshv2_levels[2] = new DTArray(local_lbound,local_extent,local_storage);


   // Allocate arrays for depth profiles
   DTArray H1(local_lbound,local_extent,local_storage);
   DTArray H2(local_lbound,local_extent,local_storage);   
   DTArray He(local_lbound,local_extent,local_storage);
   DTArray  H(local_lbound,local_extent,local_storage);
   DTArray  Hx(local_lbound,local_extent,local_storage);
   DTArray  Hy(local_lbound,local_extent,local_storage);

   // Allocate arrays for forcing
   DTArray F1x(local_lbound,local_extent,local_storage);
   DTArray F1y(local_lbound,local_extent,local_storage);
   DTArray F2x(local_lbound,local_extent,local_storage);
   DTArray F2y(local_lbound,local_extent,local_storage);

   DTArray hu1_star(local_lbound,local_extent,local_storage);
   DTArray hv1_star(local_lbound,local_extent,local_storage);
   DTArray hu2_star(local_lbound,local_extent,local_storage);
   DTArray hv2_star(local_lbound,local_extent,local_storage);
   DTArray u1_star(local_lbound,local_extent,local_storage);
   DTArray v1_star(local_lbound,local_extent,local_storage);
   DTArray u2_star(local_lbound,local_extent,local_storage);
   DTArray v2_star(local_lbound,local_extent,local_storage);


   // set equal to expression defined in pre-processing
   H1 = H1_DEPTH;
   H2 = H2_DEPTH;
   H  = H1+H2;  
   He = (H1*H2)/H;
   double Hmax = max(H);
   double Hemax = max(He);
   double c0max = sqrt(G*Hemax);

   // set initial conditions
   //*eta_levels[1] = 0;
   *eta_levels[0] = 0.5*sin(M_PI*x(ii)/Lx)*exp(-pow(((x(ii)-3.0e4)/5.0e3),2));
   //*eta_levels[0] = 0.01*(y(kk)-Ly/2)/Ly; //linear tilt
   // *eta_levels[1] = cos(M_PI*y(kk)/Ly); //half cosine (like a linear tilt)
   //*eta_levels[1] = (H0/4)*exp(-.1*((y(kk)/1e2)*(y(kk)/1e2))
   //                             -1*((x(ii)-0.5*Lx)/3e2)*((x(ii)-0.5*Lx)/3e2));

  // *u_levels[1] = 1;
    *u1_levels[0] = 0.0;
    *u2_levels[0] = 0.0;
    *v1_levels[0] = 0.0;
    *v2_levels[0] = 0.0;

    *h1_levels[0] = H1 - *eta_levels[0];
    *h2_levels[0] = H2 + *eta_levels[0];
   //*u_levels[1] = (c0max/H0)*(*eta_levels[1]);
   
   //*v_levels[1] = 0.0;

   int tstep = 0;  //time-step counter
   // Compute time-stepping details
   //double dt = SAFETY_FACTOR*fmin(Lx/Nx,Ly/Ny)/c0max;
   double dt = 1.2; // works for AB2 test case
   //double dt = 1.2/10; //fails for AB3 test case

   //double dt = 0.6;
   // this assumes uniform spacing in Cheby (y-)direction, should fix.
   double t = INITIAL_T;

   //new:
   //times[1]=t+dt; times[0]=t; times[-1]=t; times[-2]=t;

   times[1]=t+dt; times[0]=t; times[-1]=t; times[-2]=t;
   //get_coeff(times, coeffs_left, coeffs_right);
   get_ab3_coeff(times,coeffs_right);

   double numsteps = (FINAL_T - INITIAL_T)/dt;
   int outputinterval = (int) floor(numsteps/NUMOUTS);
   if (master()) printf("Using timestep of %gs, with final time of %gs\n",dt,FINAL_T);
   if (master()) printf("Reference: C_0_max = %g m/s, dx = %g m, dy = %g m\n",c0max,Lx/Nx,Ly/Ny);

   // initialize output stream for times file, and write out initial time.
   ofstream timefile;
   if (master()) {
       timefile.open("times", ios::out | ios::trunc);
       timefile << t << "\n";
       timefile.close();
   }

   // Allocate space for auxiliary elliptic variable
   DTArray z(local_lbound,local_extent,local_storage);
   DTArray p(local_lbound,local_extent,local_storage);
   DTArray dump(local_lbound,local_extent,local_storage);

   if (OUTFLAG == true)  {
       write_reader(*eta_levels[0],"eta",true);
       write_reader(*u1_levels[0],"u1",true);
       write_reader(*v1_levels[0],"v1",true);
       write_reader(*u2_levels[0],"u2",true);
       write_reader(*v2_levels[0],"v2",true);
       write_reader(H,"H");
       write_reader(p,"p",true);

       write_array(*eta_levels[0],"eta",tstep/outputinterval); 
       write_array(*u1_levels[0],"u1",tstep/outputinterval);
       write_array(*v1_levels[0],"v1",tstep/outputinterval);
       write_array(*u2_levels[0],"u2",tstep/outputinterval);
       write_array(*v2_levels[0],"v2",tstep/outputinterval);
       write_array(H,"H");

       if (NHOUTFLAG == true) 
           write_reader(z,"z",true);
   }
   
   // Allocate space for some temporaries
   DTArray temp1(local_lbound,local_extent,local_storage),
           temp2(local_lbound,local_extent,local_storage),
           temp3(local_lbound,local_extent,local_storage),
           temp4(local_lbound,local_extent,local_storage);

   // Allocate space for forcing vector
   DTArray Fx(local_lbound,local_extent,local_storage),
           Fy(local_lbound,local_extent,local_storage);

   // Allocate space for elliptic problem variable coefficients.
   DTArray cx(local_lbound,local_extent,local_storage),
           cxx(local_lbound,local_extent,local_storage),
           cy(local_lbound,local_extent,local_storage),
           cyy(local_lbound,local_extent,local_storage);

   // write grids to file
   temp1 = x(ii) + 0*kk;
   write_array(temp1,"xgrid");
   write_reader(temp1,"xgrid");
 
   temp1 = y(kk);
   write_array(temp1,"ygrid");
   write_reader(temp1,"ygrid");
  
   // Build gradient operator, calling seq: szx,szy,szz, s_exp in x,y,z
   Grad mygrad(Nx,1,Ny,FOURIER,NONE,CHEBY); // had fourier,fourier,cheby
   
   // set (constant) jacobian based on our rectangular grid. 
   mygrad.set_jac(firstDim,firstDim,2*M_PI/Lx);
   mygrad.set_jac(secondDim,secondDim,1.0);
   mygrad.set_jac(thirdDim,thirdDim,-2/Ly); //Cheby grids need the '-' 
   
   // set coefficients of elliptic problem
   cxx = (H1*H2)/3 + (H2*H2)/3;  //coeff of z_xx
   cyy = (H1*H2)/3 + (H2*H2)/3;  //coeff of z_yy
   
   //cx & cy (coefs of z_x and z_y) are the derivatives of H2o6 
   //we'll get them numerically.
   mygrad.setup_array(&cxx,FOURIER,NONE,CHEBY);
   mygrad.get_dx(&cx);  //get x derivative, store in cx
   mygrad.get_dz(&cy);  //get y derivative, store in cy 

   mygrad.setup_array(&H,FOURIER,NONE,CHEBY);
   mygrad.get_dx(&Hx);
   mygrad.get_dz(&Hy);

   //these have been validated against matlab code.
   /* write_array(cxx,"H2o6");
   write_reader(cxx,"H2o6");
   write_array(cx,"H2o6x");
   write_reader(cx,"H2o6x");
   write_array(cy,"H2o6y");
   write_reader(cy,"H2o6y"); */

   
   // create gmres-2d solver interface
   Cheb_2dmg a(Nx,Ny);          //Helmholtz solver
   Cheb_2dmg lapsolver(Nx,Ny);  //Poisson solver
  
   int itercount; //gmres iteration count  
   double zdbc=0.0;
   double znbc=-1.0;
   double xdbc=0.0;
   double xnbc=0.0;  

   // set Helmholtz variable coefs (must do before set_grad)
   a.set_ci(&cx,1);  //coeff of zx
   a.set_ci(&cxx,2); //coeff of zxx
   a.set_ci(&cy,3);  //coeff of zy
   a.set_ci(&cyy,4); //coeff of zyy

   a.set_grad(&mygrad); // need to do this before set_bc 

   // set_bc, calling sequence is helm, dir., neu., S_EXP ('x' series expans)
   //old version was a.set_bc(-1.0,dbc,nbc,FOURIER);
   a.set_bc(-1.0,zdbc,znbc,FOURIER,xdbc,xnbc); // Sign of 'helm' opposite that of 1D code.  %trailing zeros new as of Aug. 3, 2013.
   //may be setting this up wrong, ask the Subich-man

   lapsolver.set_ci(&Hx,1); // coeff of px
   lapsolver.set_ci(&H,2);  // coeff of pxx
   lapsolver.set_ci(&Hy,3); // coeff of py
   lapsolver.set_ci(&H,4);  // coeff of pyy

   lapsolver.set_grad(&mygrad);
   lapsolver.set_bc(0.0,zdbc,znbc,FOURIER,xdbc,xnbc); //Helmholtz = 0.0

   // Initialize structures that are used by gmres object
   fbox * init_r_helm = a.alloc_resid();
   ubox * final_u_helm = a.alloc_basis();

   fbox * init_r_pois = lapsolver.alloc_resid();
   ubox * final_u_pois =lapsolver.alloc_basis();
   
   //Build the solver object with template class
   GMRES_Solver<Cheb_2dmg> mysolverHelm(&a);
   GMRES_Solver<Cheb_2dmg> mysolverPois(&lapsolver);

   // set forcing
   F1x = 0;
   F1y = 0;
   F2x = 0;
   F2y = 0;

   double starttime = MPI_Wtime();

   // Main time-stepping loop follows
   if (master()) printf("Entering main loop... (Adams-Bashforth)\n");
   while (t < FINAL_T) {
      //make nice references for arrays used
      DTArray &u1_n = *u1_levels[0], &v1_n = *v1_levels[0],
              &u2_n = *u2_levels[0], &v2_n = *v2_levels[0],
              &hu1_n = *hu1_levels[0], &hv1_n = *hv1_levels[0],
              &hu2_n = *hu2_levels[0], &hv2_n = *hv2_levels[0],             
 
              &eta_n = *eta_levels[0],
              &h1_n  = *h1_levels[0],
              &h2_n  = *h2_levels[0],
            
              &u1_np1 = *u1_levels[1], &v1_np1 = *v1_levels[1], 
              &u2_np1 = *u2_levels[1], &v2_np1 = *v2_levels[1],
              &hu1_np1= *hu1_levels[1], &hv1_np1 = *hv1_levels[1],
              &hu2_np1= *hu2_levels[1], &hv2_np1 = *hv2_levels[1], 

              &eta_np1 = *eta_levels[1],
              &h1_np1  = *h1_levels[1],
              &h2_np1  = *h2_levels[1],

              &rhsh2_nm2 = *rhsh2_levels[0],
              &rhsh2_nm1 = *rhsh2_levels[1],
              &rhsh2_n   = *rhsh2_levels[2],

              &rhshu1_nm2= *rhshu1_levels[0],
              &rhshu1_nm1= *rhshu1_levels[1],
              &rhshu1_n  = *rhshu1_levels[2],
              &rhshv1_nm2= *rhshv1_levels[0],
              &rhshv1_nm1= *rhshv1_levels[1],
              &rhshv1_n  = *rhshv1_levels[2],

              &rhshu2_nm2= *rhshu2_levels[0],
              &rhshu2_nm1= *rhshu2_levels[1],
              &rhshu2_n  = *rhshu2_levels[2],
              &rhshv2_nm2= *rhshv2_levels[0],
              &rhshv2_nm1= *rhshv2_levels[1],
              &rhshv2_n  = *rhshv2_levels[2],

              &temp = temp1,
              &a1x = temp2, &a2y = temp3,
              &rhs = temp4;
              //&eta_rhs_nm1 = temp3, &eta_rhs_nm2 = temp4,
              //&a1 = temp4, &a2 = temp5;
              
       //update forcing:
       F1x = 0;
       F1y = 0;
       F2x = 0;
       F2y = 0;

       //step eta:
       //coeffs_right
       //eta_p = eta_m - 2*dt*((eta_n u)_x + (eta_n v)_y)
       temp = -hu2_n;
       mygrad.setup_array(&temp,FOURIER,NONE,CHEBY);
       mygrad.get_dx(&rhsh2_n,false);  //get x derivative, store in rhs
       temp = -hv2_n;
       mygrad.setup_array(&temp,FOURIER,NONE,CHEBY);
       mygrad.get_dz(&rhsh2_n,true); //this time, true = we sum with old result.

       h2_np1 = h2_n + dt*(coeffs_right[1]*rhsh2_n 
                            +coeffs_right[0]*rhsh2_nm1 
                            +coeffs_right[-1]*rhsh2_nm2);

       eta_np1 = h2_np1 - H2;
       h1_np1 = H - h2_np1;
       

       // filter free surface
       if (FILTER_ON == true)
         filter3(eta_np1,XY_xform,FOURIER,NONE,CHEBY,F_CUTOFF,F_ORDER,F_STREN);

       //impose explicit Neuman conditions on eta.
      double neurhs1 = 0.0;
       double neurhs2 = 0.0;
       for (int j=local_lbound(0); j<(local_lbound(0)+local_extent(0)); j++) {
           neurhs1 = sum(-Dy(0,Range(1,Ny-2))*eta_np1(j,0,Range(1,Ny-2)));
           neurhs2 = sum(-Dy(Ny-1,Range(1,Ny-2))*eta_np1(j,0,Range(1,Ny-2)));

           // perform 2x2 matrix product at each point 
           eta_np1(j,0,0)   = (neumatinv(0,0)*neurhs1 + neumatinv(0,1)*neurhs2);
           eta_np1(j,0,Ny-1) = (neumatinv(1,0)*neurhs1+ neumatinv(1,1)*neurhs2);
       }  //this code is right.

       //compute h1np1 and h2np1
       h2_np1 = H2+eta_np1;
       h1_np1 = H-h2_np1; //this line ensures no errors in mass conservation.
       //h1_np1 = H1-eta_np1;

       //Layer 1, U eqn:
       temp = -hu1_n*u1_n;
       mygrad.setup_array(&temp,FOURIER,NONE,CHEBY);
       mygrad.get_dx(&rhshu1_n,false);
       temp = -hu1_n*v1_n;
       mygrad.setup_array(&temp,FOURIER,NONE,CHEBY);
       mygrad.get_dz(&rhshu1_n,true);
       //add coriolis and forcing
       rhshu1_n = rhshu1_n + (ROT_F+ROT_B*y(kk))*hv1_n + h1_n*F1x;
       
       //Layer 1, V eqn:
       temp = -hv1_n*u1_n;
       mygrad.setup_array(&temp,FOURIER,NONE,CHEBY);
       mygrad.get_dx(&rhshv1_n,false);
       temp = -hv1_n*v1_n;
       mygrad.setup_array(&temp,FOURIER,NONE,CHEBY);
       mygrad.get_dz(&rhshv1_n,true);
       //add coriolis and forcing
       rhshv1_n = rhshv1_n - (ROT_F+ROT_B*y(kk))*hu1_n + h2_n*F1y;

       //Layer 2, U eqn:
       temp = -hu2_n*u2_n;
       mygrad.setup_array(&temp,FOURIER,NONE,CHEBY);
       mygrad.get_dx(&rhshu2_n,false);
       temp = -hu2_n*v2_n;
       mygrad.setup_array(&temp,FOURIER,NONE,CHEBY);
       mygrad.get_dz(&rhshu2_n,true);
       temp = eta_n;
       mygrad.setup_array(&temp,FOURIER,NONE,CHEBY);
       mygrad.get_dx(&dump,false);
       //add coriolis and forcing
       rhshu2_n = rhshu2_n - (ROT_F+ROT_B*y(kk))*hv2_n + h2_n*F1x;
       //add interfacial pressure gradient
       rhshu2_n = rhshu2_n - G*(h2_n*dump);

       //Layer 2, V eqn:
       temp = -hu2_n*v2_n;
       mygrad.setup_array(&temp,FOURIER,NONE,CHEBY);
       mygrad.get_dx(&rhshv2_n,false);
       temp = -hv2_n*v2_n;
       mygrad.setup_array(&temp,FOURIER,NONE,CHEBY);
       mygrad.get_dz(&rhshv2_n,true);
       temp = eta_n;
       mygrad.setup_array(&temp,FOURIER,NONE,CHEBY);
       mygrad.get_dz(&dump,false);
       //add coriolis and forcing
       rhshv2_n = rhshv2_n - (ROT_F+ROT_B*y(kk))*hu2_n + h2_n*F1y;
       //add interfacial pressure gradient
       rhshv2_n = rhshv2_n - G*(h2_n*dump);

       //form a1x and a2y:
       temp = rhshu2_n; //a1
       mygrad.setup_array(&temp,FOURIER,NONE,CHEBY);
       mygrad.get_dx(&a1x,false);
       
       temp = rhshv2_n;  //a2
       mygrad.setup_array(&temp,FOURIER,NONE,CHEBY);
       mygrad.get_dz(&a2y,false);

       //compute - divergence of \vec{a}, store in rhs for helmholtz problem
       rhs = -(a1x+a2y);

       //set bc's on RHS - TODO: there might be a sign error on one of the BC's
       rhs(Range::all(),0,0)   =-rhshv2_n(Range::all(),0,0)/cxx(Range::all(),0,0);
       rhs(Range::all(),0,Ny-1)= rhshv2_n(Range::all(),0,Ny-1)/cxx(Range::all(),0,Ny-1);  //recent sign change.

       //solve for z (non-hydrostatic pressure) with gmres
       *init_r_helm->gridbox = rhs;

       itercount = mysolverHelm.Solve(final_u_helm,init_r_helm,TOL,MAXIT,RESTARTS);

       if (master() && NOISYGMRES == true)
           cout << "GMRES converged after " << itercount << " iterations." << endl;

       z = *final_u_helm->gridbox;


       // cout << "z:" << z << endl;
 
       // Do actual time-stepping of hyperbolic terms
       hu1_star = hu1_n + dt*(coeffs_right[1]*rhshu1_n
                            +coeffs_right[0]*rhshu2_nm1
                            +coeffs_right[-1]*rhshu2_nm2);

       hv1_star = hv1_n + dt*(coeffs_right[1]*rhshv1_n
                            +coeffs_right[0]*rhshv1_nm1
                            +coeffs_right[-1]*rhshv2_nm2);
     
       hu2_star = hu2_n + dt*(coeffs_right[1]*rhshu2_n
                            +coeffs_right[0]*rhshu2_nm1
                            +coeffs_right[-1]*rhshu2_nm2);

       hv2_star = hv2_n + dt*(coeffs_right[1]*rhshv2_n
                            +coeffs_right[0]*rhshv2_nm1
                            +coeffs_right[-1]*rhshv2_nm2);
 
       // cout << "hv2_star:" << hv2_star << endl;

       //compute gradient of NH pressure, then time-step it in lower layer
       mygrad.setup_array(&z,FOURIER,NONE,CHEBY);
       mygrad.get_dx(&temp,false); // z_x
       hu2_star = hu2_star + dt*(cxx*temp); //cxx = NH Coeff
       mygrad.get_dz(&temp,false); // z_y
       hv2_star = hv2_star + dt*(cxx*temp); //cxx = NH Coeff

       //recover predicted velocities
       u1_star = hu1_star / h1_np1;
       v1_star = hv1_star / h1_np1;
       u2_star = hu2_star / h2_np1;
       v2_star = hv2_star / h2_np1;

       //filter predicted velocities
       if (FILTER_ON == true) {
         filter3(u1_star,XY_xform,FOURIER,NONE,CHEBY,F_CUTOFF,F_ORDER,F_STREN);
         filter3(v1_star,XY_xform,FOURIER,NONE,CHEBY,F_CUTOFF,F_ORDER,F_STREN);
         filter3(u2_star,XY_xform,FOURIER,NONE,CHEBY,F_CUTOFF,F_ORDER,F_STREN);
         filter3(v2_star,XY_xform,FOURIER,NONE,CHEBY,F_CUTOFF,F_ORDER,F_STREN);
       }
   
       //Re-form predicted transports for projection step
       hu1_star = u1_star * h1_np1;
       hv1_star = v1_star * h1_np1;
       hu2_star = u2_star * h2_np1;
       hv2_star = v2_star * h2_np1;

       //form hustartot and compute divergence
       temp = (hu1_star + hu2_star); //a1
       mygrad.setup_array(&temp,FOURIER,NONE,CHEBY);
       mygrad.get_dx(&a1x,false);

       temp = (hv1_star + hv2_star); //a2
       // cout << temp << endl;
       mygrad.setup_array(&temp,FOURIER,NONE,CHEBY);
       mygrad.get_dz(&a2y,false);

       //if (master())
       //     cout << a2y << endl;     

       //compute - divergence of \vec{a}, store in rhs for helmholtz problem
       rhs = (1/dt)*(a1x+a2y);

       //set bc's on RHS -- temp contains hvtot_star 
       temp = hv1_star + hv2_star;
       rhs(Range::all(),0,0)   = (1/dt)*temp(Range::all(),0,0);
       rhs(Range::all(),0,Ny-1)= - (1/dt)*temp(Range::all(),0,Ny-1); //recent sign change


       //solve for z (hydrostatic pressure) with gmres
       *init_r_pois->gridbox = rhs;

       itercount = mysolverPois.Solve(final_u_pois,init_r_pois,TOL,MAXIT,RESTARTS);
       if (master() && NOISYGMRES == true)
           cout << "GMRES converged after " << itercount << " iterations." << endl;

       p = *final_u_pois->gridbox;

       // cout << "p:" << p << endl;

       //compute gradient of pressure, then time-step it in each layer
       mygrad.setup_array(&p,FOURIER,NONE,CHEBY);
       mygrad.get_dx(&temp,false); // p_x
       hu1_np1 = hu1_star - dt*(h1_np1*temp);
       hu2_np1 = hu2_star - dt*(h2_np1*temp); 

       mygrad.get_dz(&temp,false); // p_y
       hv1_np1 = hv1_star - dt*(h1_np1*temp);
       hv2_np1 = hv2_star - dt*(h2_np1*temp);

       // cout << "hv2_np1: " << hv2_np1 << endl;
       // cout << "hv1_np1: " << hv1_np1 << endl;

       //return 0;

       //Recover v field & u field
       v1_np1 = hv1_np1 / h1_np1;
       v2_np1 = hv2_np1 / h2_np1;

       u1_np1 = hu1_np1 / h1_np1;
       u2_np1 = hu2_np1 / h2_np1;

       //Impose BCs on v at y=0,L_y (Dirichlet -> no normal flow)
       v1_np1(Range::all(),0,0) =0;
       v1_np1(Range::all(),0,Ny-1) = 0;
       
       v2_np1(Range::all(),0,0) =0;
       v2_np1(Range::all(),0,Ny-1) = 0;

      
       //filter corrected velocities
       if (FILTER_ON == true) {
         filter3(u1_np1,XY_xform,FOURIER,NONE,CHEBY,F_CUTOFF,F_ORDER,F_STREN);
         filter3(v1_np1,XY_xform,FOURIER,NONE,CHEBY,F_CUTOFF,F_ORDER,F_STREN);
         filter3(u2_np1,XY_xform,FOURIER,NONE,CHEBY,F_CUTOFF,F_ORDER,F_STREN);
         filter3(v2_np1,XY_xform,FOURIER,NONE,CHEBY,F_CUTOFF,F_ORDER,F_STREN);
       }
   
    
       
       //Re-form transports
       hv1_np1 = h1_np1*v1_np1;
       hv2_np1 = h2_np1*v2_np1;

       hu1_np1 = h1_np1*u1_np1;
       hu2_np1 = h2_np1*u2_np1;

       //increment time.
       tstep++;
       t+=dt;
       //cout << t << endl;

       if (TIMESTEPPER == AB3){
           //shift down 'stepped' times array by 1, and set new time.
           //AB3
       	   times[-2] = times[-1];
           times[-1] = times[0];
           times[0] = times[1];
           times[1] = t+dt;
       }
       else if (TIMESTEPPER == AB2){
           //AB2
           times[-2] = times[0];
           times[-1] = times[0];
           times[0]=times[1];
           times[1]=t+dt;
       }
       else {
           if (master())
               cout << "Invalid timestepper, exiting..." << endl;
               return 0;
       }
      
       // cout << times; 
       get_ab3_coeff(times,coeffs_right);

       // Check if it's time to output
       if(!(tstep % outputinterval) && OUTFLAG == true){
           if (master())
               cout << "outputting at t=" << t << "\n";

           write_array(*eta_levels[1],"eta",tstep/outputinterval);
           write_array(*u1_levels[1],"u1",tstep/outputinterval);
           write_array(*u2_levels[1],"u2",tstep/outputinterval);
           write_array(*v1_levels[1],"v1",tstep/outputinterval);
           write_array(*v2_levels[1],"v2",tstep/outputinterval);
           write_array(p,"p",tstep/outputinterval);

           if (NHOUTFLAG == true) 
               write_array(z,"z",tstep/outputinterval);
          
           //append current time to times file
           if (master()) {
               timefile.open ("times", ios::out | ios::app);
               timefile << t << "\n";
               timefile.close();
           }
       }  // end output 'if'

       //generic log-style text output to screen
       if(!(tstep % 100) || tstep < 20) {
           // cout << "test" << endl;
           if (master()) printf("Completed time %g (tstep %d)\n",t,tstep);
           double mu1 = psmax(max(abs(u1_np1))), mv1 = psmax(max(abs(v1_np1))), meta = pvmax(eta_np1);
           if (master()) {
                printf("Max u1 %g, v1 %g, eta %g\n",mu1,mv1,meta);
                if (isnan(mu1)) {
                    printf("NaNs detected, terminating...");
                    return 0;
                }
           }
                
       }

       //cycle fields for next timestep by shuffling pointers.
       DTArray * tmp ;
       tmp = u1_levels[0];
       u1_levels[0] = u1_levels[1];
       u1_levels[1] = tmp;

       tmp = u2_levels[0];
       u2_levels[0] = u2_levels[1];
       u2_levels[1] = tmp;

       tmp = hu1_levels[0];
       hu1_levels[0] = hu1_levels[1];
       hu1_levels[1] = tmp;

       tmp = hu2_levels[0];
       hu2_levels[0] = hu2_levels[1];
       hu2_levels[1] = tmp;
      
       tmp = v1_levels[0];
       v1_levels[0] = v1_levels[1];
       v1_levels[1] = tmp;

       tmp = v2_levels[0];
       v2_levels[0] = v2_levels[1];
       v2_levels[1] = tmp;

       tmp = hv1_levels[0];
       hv1_levels[0] = hv1_levels[1];
       hv1_levels[1] = tmp;

       tmp = hv2_levels[0];
       hv2_levels[0] = hv2_levels[1];
       hv2_levels[1] = tmp;

       tmp = eta_levels[0];
       eta_levels[0] = eta_levels[1];
       eta_levels[1] = tmp;

       tmp = h1_levels[0];
       h1_levels[0] = h1_levels[1];
       h1_levels[1] = tmp;
 
       tmp = h2_levels[0];
       h2_levels[0] = h2_levels[1];
       h2_levels[1] = tmp;
       
       tmp = rhsh2_levels[0];
       rhsh2_levels[0] = rhsh2_levels[1];
       rhsh2_levels[1] = rhsh2_levels[2];
       rhsh2_levels[2] = tmp;

       tmp = rhshu1_levels[0];
       rhshu1_levels[0] = rhshu1_levels[1];
       rhshu1_levels[1] = rhshu1_levels[2];
       rhshu1_levels[2] = tmp;
       
       tmp = rhshv1_levels[0];
       rhshv1_levels[0] = rhshv1_levels[1];
       rhshv1_levels[1] = rhshv1_levels[2];
       rhshv1_levels[2] = tmp;

       tmp = rhshu2_levels[0];
       rhshu2_levels[0] = rhshu2_levels[1]; 
       rhshu2_levels[1] = rhshu2_levels[2];
       rhshu2_levels[2] = tmp;

       tmp = rhshv2_levels[0];
       rhshv2_levels[0] = rhshv2_levels[1];
       rhshv2_levels[1] = rhshv2_levels[2];
       rhshv2_levels[2] = tmp;

 
   } //end Leapfrog while loop
   
   double endtime = MPI_Wtime();
   double runtime = endtime-starttime;
   if (master()) {
        printf("Finished run. Run-time = %g s \n", runtime);
        timefile.open ("runtime", ios::out | ios::trunc);
        timefile << runtime << "\n";
        timefile.close();
   }
   
   // free memory of residual and basis of gmres solver 
   a.free_resid(init_r_helm);
   a.free_basis(final_u_helm);

   lapsolver.free_resid(init_r_pois);
   lapsolver.free_basis(final_u_pois);

    
   MPI_Finalize();
   return 0;
}


