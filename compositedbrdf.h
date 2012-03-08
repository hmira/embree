// ======================================================================== //
// Copyright 2009-2011 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#ifndef __EMBREE_COMPOSITED_BRDF_H__
#define __EMBREE_COMPOSITED_BRDF_H__

#include "brdfs/brdf.h"
#include <stdio.h>

/*! Helper makro that allocates memory in the composited BRDF and
 *  performs an inplace new of the BRDF to create. */
#ifndef NEW_BRDF
#define NEW_BRDF(...) new (brdfs.alloc(sizeof(__VA_ARGS__))) __VA_ARGS__
#endif

namespace embree
{
  /*! Composited BRDF deals as container of individual BRDF
   *  components. It contains storage where its BRDF components are
   *  allocated inside. It has to be aligned, because the BRDF
   *  components might use SSE code. */
  class __align(16) CompositedBRDF
  {
    /*! maximal number of BRDF components */
    enum { maxComponents = 8 };

    /*! maximal number of bytes for BRDF storage */
    enum { maxBytes = 256*sizeof(float) };

  public:

    /*! Composited BRDF constructor. */
    __forceinline CompositedBRDF() : numBytes(0), numBRDFs(0) {}

    /*! Allocates data for new BRDF component. Data gets aligned by 16
     *  bytes. */
    __forceinline void* alloc(size_t size) {
      assert(numBytes+size<maxBytes);
      void* p = &data[numBytes];
      numBytes = (numBytes+size+15)&(size_t(-16));
      return p;
    }

    /*! Adds a new BRDF to the list of BRDFs */
    __forceinline void add(const BRDF* brdf) {
      assert(numBRDFs < 8);
      if (numBRDFs < 8) BRDFs[numBRDFs++] = brdf;
    }

    /*! Returns the number of used BRDF components. */
    __forceinline size_t size() const { return numBRDFs; }

    /*! Returns a pointer to the i'th BRDF component. */
    __forceinline const BRDF* operator[] (size_t i) const { return BRDFs[i]; }

    /*! Evaluates all BRDF components. */
    Col3f eval(const Vec3f& wo, const DifferentialGeometry& dg, const Vec3f& wi, BRDFType type) const
    {
      Col3f c = zero;
      for (size_t i=0; i<size(); i++)
        if (BRDFs[i]->type & type) c += BRDFs[i]->eval(wo,dg,wi);
      return c;
    }

    
    float pdf(const Vec3f               & wo,          /*!< Direction light is reflected into.                    */
                 const DifferentialGeometry& dg,          /*!< Shade location on a surface to sample the BRDF at.    */
                 const Sample3f                  & wi,        /*!< Returns sampled incoming light direction and PDF.     */
                 float                       ss,          /*!< Sample to select the BRDF component.                  */
                 const BRDFType            & type = ALL)  /*!< The type of BRDF components to consider.              */ const
    {
      size_t num = 0;
      float PDFs[maxComponents];
      float colors[maxComponents];
      float f[maxComponents];
      float refl[maxComponents];
      float total = 0.0f;
      float sum = 0.0f;
      
      float subtotal = 0.0f;
      for (size_t i = 0; i<size(); i++)
      {
        if (!(BRDFs[i]->type & type)) continue;
        PDFs[i] = BRDFs[i]->pdf(wo, dg, wi);
	Col3f c = BRDFs[i]->eval(wo, dg, wi);
	sum += f[i] = reduce_max(c);
	total += PDFs[i];
	subtotal += refl[i] * PDFs[i];
	num++;
      }
      
      /*! normalize distribution */
      for (size_t i = 0; i<num; i++) f[i] /= sum;

      /*! compute accumulated distribution */
      float d[maxComponents];
      d[0] = f[0];
      for (size_t i=1; i<num-1; i++) d[i] = d[i-1] + f[i];
      d[num-1] = 1.0f;
      
      /*! sample distribution */
      size_t i = 0; while (i<num-1 && ss > d[i]) i++;
      
      if (BRDFs[i]->type & GLOSSY) return 0.0f;
      
      return PDFs[i] * f[i];
    }
    
    
    
    
    /*! Sample the composited BRDF. We are evaluating all BRDF
     *  components and then importance sampling one of them. */
    Col3f sample(const Vec3f               & wo,          /*!< Direction light is reflected into.                    */
                 const DifferentialGeometry& dg,          /*!< Shade location on a surface to sample the BRDF at.    */
                 Sample3f                  & wi_o,        /*!< Returns sampled incoming light direction and PDF.     */
                 BRDFType                  & type_o,      /*!< Returns the type flags of samples component.          */
                 const Vec2f               & s,           /*!< Sample locations for BRDF are provided by the caller. */
                 float                       ss,          /*!< Sample to select the BRDF component.                  */
                 const BRDFType            & type = ALL)  /*!< The type of BRDF components to consider.              */ const
    {
      /*! probability distribution to sample between BRDF components */
      float f[maxComponents];
      float sum = 0.0f;

      /*! stores sampling of each BRDF component */
      Col3f colors[maxComponents];
      Sample3f samples[maxComponents];
      BRDFType types[maxComponents];
      size_t num = 0;

      /*! sample each BRDF component and build probability distribution */
      for (size_t i = 0; i<size(); i++)
      {
        if (!(BRDFs[i]->type & type)) continue;
        Sample3f wi; Col3f c = BRDFs[i]->sample(wo, dg, wi, s);//printf("%5.2f\n",BRDFs[i]->pdf(wo, dg, wi));
        if (c == Col3f(zero) || wi.pdf <= 0.0f) continue;
        //sum += f[num] = (c.r + c.g + c.b) * rcp(wi.pdf);
	sum += f[num] = reduce_max(c);
	colors[num] = c;
        samples[num] = wi;
        types[num] = BRDFs[i]->type;
        num++;
      }

      /*! exit if we did not find any valid component */
      if (num == 0) {
        wi_o = Sample3f(zero,0.0f);
        type_o = (BRDFType)0;
        return zero;
      }

      /*! normalize distribution */
      for (size_t i = 0; i<num; i++) f[i] /= sum;

      /*! compute accumulated distribution */
      float d[maxComponents];
      d[0] = f[0];
      for (size_t i=1; i<num-1; i++) d[i] = d[i-1] + f[i];
      d[num-1] = 1.0f;

      /*! sample distribution */
      size_t i = 0; while (i<num-1 && ss > d[i]) i++;

      /*! return */
      wi_o = Sample3f(samples[i].value,samples[i].pdf*f[i]);
      type_o = types[i];
      return colors[i];
    }

  private:

    /*! Data storage. Has to be at the beginning of the class due to alignment. */
    char data[maxBytes];               //!< Storage for BRDF components
    size_t numBytes;                   //!< Number of bytes occupied in storage

    /*! BRDF list */
    const BRDF* BRDFs[maxComponents]; //!< pointers to BRDF components
    size_t numBRDFs;                  //!< number of stored BRDF components
  };
}

#endif
