#include <blitz/array.h>
#include <math.h>

using blitz::Array;
using blitz::TinyVector;
using blitz::GeneralArrayStorage;
using blitz::Range;
using blitz::firstDim;
using blitz::secondDim;

// Blitz index placeholders

//blitz::firstIndex ii;
//blitz::secondIndex jj;
//blitz::thirdIndex kk;

Array<double,2> chebmat(int Nx) {

    blitz::firstIndex ii;
    blitz::secondIndex jj;
    
    Array<double,1> x(Nx);
    x = cos(M_PI*ii/(Nx-1));
    
   Array<double,2> myI(Nx,Nx);
   Array<double,2> Dx(Nx,Nx);
   Array<double,2> myX(Nx,Nx);
   Array<double,1> myc(Nx);
   Array<double,2> mydX(Nx,Nx);
   Array<double,2> mysum(Nx,Nx);

   // build identity matrix (any way to clean this up?)
   for (int j=0; j<Nx; j++)
       myI(j,j) = 1.0;

   myc(0) = 2.0;
   myc(Range(1,Nx-2)) = 1.0;
   myc(Nx-1) = 2.0;
   myc = myc*pow(-1.0,ii);

   // like the "repmat" step - clean this up with blitz functionality?
   for (int j=0; j<Nx; j++)
        myX(Range::all(),j) = x;

   mydX = myX-myX.transpose(secondDim,firstDim);
   Dx = myc(ii)*(1/myc(jj));  //outer product
   Dx = Dx/(mydX + myI);

   // build diagonal matrix of row-sums
   // perhaps try: sum(Dx(i,Range::all()))
   double rowsum;
   for (int i=0; i<Nx; i++) {
       rowsum = 0.0;
       for (int j=0; j<Nx; j++) {
           rowsum+= Dx(i,j);
       }
       mysum(i,i) = rowsum;
   }

   Dx = Dx - mysum;

   return Dx;
}
