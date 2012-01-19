#define _CRT_RAND_S

#include "integrators/pathtraceintegrator.h"

namespace embree
{
  
  inline float mis ( float pdf1, float pdf2 )
  {
    return pdf1 / (pdf1 + pdf2);
  }
  
  
  
    
  
  
  PathTraceIntegrator::PathTraceIntegrator(const Parms& parms)
    : lightSampleID(-1), firstScatterSampleID(-1), firstScatterTypeSampleID(-1)
  {
    maxDepth        = parms.getInt  ("maxDepth"       ,10    );
    minContribution = parms.getFloat("minContribution",0.01f );
    epsilon         = parms.getFloat("epsilon"        ,128.0f)*float(ulp);
    backplate       = parms.getImage("backplate");
  }

  void PathTraceIntegrator::requestSamples(Ref<SamplerFactory>& samplerFactory, const Ref<BackendScene>& scene)
  {
    precomputedLightSampleID.resize(scene->allLights.size());

    lightSampleID = samplerFactory->request2D();
    for (size_t i=0; i<scene->allLights.size(); i++) {
      precomputedLightSampleID[i] = -1;
      if (scene->allLights[i]->precompute())
        precomputedLightSampleID[i] = samplerFactory->requestLightSample(lightSampleID, scene->allLights[i]);
    }
    firstScatterSampleID = samplerFactory->request2D((int)maxDepth);
    firstScatterTypeSampleID = samplerFactory->request1D((int)maxDepth);
  }

  Col3f PathTraceIntegrator::Li(const LightPath& lightPathOrig, const Ref<BackendScene>& scene, Sampler* sampler, size_t& numRays)
  {
	  bool done = false;
	  Col3f coeff = Col3f(1,1,1);
	  Col3f Lsum = zero;
	  Col3f L = zero;
	  LightPath lightPath = lightPathOrig;
	  bool doneDiffuse = false;

	while (!done)
	{
	  
    BRDFType directLightingBRDFTypes = (BRDFType)(DIFFUSE);
    BRDFType giBRDFTypes = (BRDFType)(ALL);

    /*! Terminate path if too long or contribution too low. */
	L = zero;
    if (lightPath.depth >= maxDepth)// || reduce_max(lightPath.throughput) < minContribution)
		return Lsum;

    /*! Traverse ray. */
    DifferentialGeometry dg;
    scene->accel->intersect(lightPath.lastRay,dg);
    scene->postIntersect(lightPath.lastRay,dg);
    const Vec3f wo = -lightPath.lastRay.dir;
    numRays++;
		float sum = 0;

    /*! Environment shading when nothing hit. */
    if (!dg)
    {
      if (backplate && lightPath.unbend) {
        Vec2f raster = sampler->getPrimary();
        int width = sampler->getImageSize().x;
        int height = sampler->getImageSize().y;
        int x = (int)((raster.x / width) * backplate->width);
        x = clamp(x, 0, int(backplate->width)-1);
        int y = (int)((raster.y / height) * backplate->height);
        y = clamp(y, 0, int(backplate->height)-1);
        L = backplate->get(x, y);
      }
      else {
        if (!lightPath.ignoreVisibleLights)
          for (size_t i=0; i<scene->envLights.size(); i++)
            L += scene->envLights[i]->Le(wo);
      }
      return Lsum + L*coeff;
    }

    /*! Shade surface. */
    CompositedBRDF brdfs;
    if (dg.material) dg.material->shade(lightPath.lastRay, lightPath.lastMedium, dg, brdfs);

    /*! face forward normals */
    bool backfacing = false;
#if defined(__EMBREE_CONSISTENT_NORMALS__) && __EMBREE_CONSISTENT_NORMALS__ > 1
    return Col3f(abs(dg.Ns.x),abs(dg.Ns.y),abs(dg.Ns.z));
#else
    if (dot(dg.Ng, lightPath.lastRay.dir) > 0) {
      backfacing = true; dg.Ng = -dg.Ng; dg.Ns = -dg.Ns;
    }
#endif
Col3f d = zero;
float ss = 0.0f;
// doneDiffuse = true;

    /*! Add light emitted by hit area light source. */
    if (!lightPath.ignoreVisibleLights && dg.light && !backfacing)
      L += dg.light->Le(dg,wo);

    /*! Check if any BRDF component uses direct lighting. */
    bool useDirectLighting = false;
    for (size_t i=0; i<brdfs.size(); i++)
      useDirectLighting |= (brdfs[i]->type & directLightingBRDFTypes) != NONE;

    /*! Direct lighting. Shoot shadow rays to all light sources. */
    if (useDirectLighting)
    {
      
      
      
      
		std::vector<float> accumRad;
		
		/*! Run through all the lightsources and sample or compute the distribution function for rnd gen */
		for (size_t i=0; i<scene->allLights.size(); i++)
      {
        /*! Either use precomputed samples for the light or sample light now. */
        LightSample ls;
        if (scene->allLights[i]->precompute()) ls = sampler->getLightSample(precomputedLightSampleID[i]);
        else ls.L = scene->allLights[i]->sample(dg, ls.wi, ls.tMax, sampler->getVec2f(lightSampleID));

	
	/*BRDF*/
		Sample3f wi_1; BRDFType type;
		Vec2f s_1  = sampler->getVec2f(firstScatterSampleID + lightPath.depth);
		ss = sampler->getFloat(firstScatterTypeSampleID + lightPath.depth);
		d = brdfs.sample(wo, dg, wi_1, type, s_1, ss, giBRDFTypes);


 		float weight1 = 0.0f;

		Col3f red = Col3f(1.0f,0.0f,0.0f);
		Col3f blue = Col3f(0.0f,0.0f,1.0f);
		if ( wi_1.pdf > 0.0f )
		{

			Ray r( dg.P, wi_1.value, dg.error*epsilon, ls.tMax-dg.error*epsilon );
			DifferentialGeometry diff;
			scene->accel->intersect( r, diff );
			scene->postIntersect( r, diff );

			Col3f radiance =  Col3f( 0.0f,0.0f,0.0f );
			if ( diff.light  )
			  radiance = diff.light->Le(diff, -wi_1.value );

 			if ( diff.light  )
 			  weight1 = mis( wi_1.pdf, diff.light->pdf(diff, wi_1.value));
			
 		  	if ( dot( diff.Ng, -r.dir ) > 0 )
 				L += radiance * d * weight1 / wi_1.pdf;

//     		  	if ( dot( diff.Ng, -r.dir ) > 0 )
//   				L += radiance * d / wi_1.pdf;

			  
			  
		}
	/*potialto*/
	
	
		/*! Start using only one random lightsource after first Lambertian reflection */
		if (doneDiffuse)
		{
			/*! run through all the lighsources and compute radiance accumulatively */
			sum += reduce_max(scene->allLights[i]->eval(dg,ls.wi));
			accumRad.push_back(sum);
		}
		else
		{
			/*! Ignore zero radiance or illumination from the back. */
			if (ls.L == Col3f(zero) || ls.wi.pdf == 0.0f || dot(dg.Ns,Vec3f(ls.wi)) <= 0.0f) continue;

			/*! Test for shadows. */
			bool inShadow = scene->accel->occluded(Ray(dg.P, ls.wi, dg.error*epsilon, ls.tMax-dg.error*epsilon));
			numRays++;
			if (inShadow) continue;

			/*! Evaluate BRDF. */
			//TODO skúsim bez lightsampling
		/*LIGHTSOURCE SAMPLING*/	
			float weight2 = 1.0f;
 			weight2 = mis( ls.wi.pdf, brdfs.pdf( wo, dg, wi_1  ));
   			L += ls.L * brdfs.eval(wo, dg, ls.wi, directLightingBRDFTypes) * rcp(ls.wi.pdf) * weight2;
//  			L += ls.L * brdfs.eval(wo, dg, ls.wi, directLightingBRDFTypes) * rcp(ls.wi.pdf);
// 			L += sampleLight( lightPath, scene, sampler, dg, brdfs, wi_1, 0 );
// 			brdfs.pdf(wi_1,dg,wo);
// L /= 2.0f;
//1

			L *= rcp(weight1 + weight2);

//-1			
			
		}
      }

	  /*! After fisrt Lambertian reflection pick one random lightsource and compute contribution */
	  if (doneDiffuse && scene->allLights.size() != 0)
	  {
		  /*! Generate the random value */
		  unsigned int RndVal;
		  (rand_r(&RndVal)); // std::cout << "\nRND gen error!\n";
		  float rnd((float)RndVal/(float)UINT_MAX);
		  
		  /*! Pick the particular lightsource according the radiosity-given distribution */
		  size_t i = 0;
		  while (i < scene->allLights.size() && rnd > accumRad[i]/sum)
			  ++i;

		  /*! Sample the selected lightsource and compute contribution */
		  if ( i >= scene->allLights.size() ) i = scene->allLights.size() -1;
		  LightSample ls;
		  if (scene->allLights[i]->precompute()) ls = sampler->getLightSample(precomputedLightSampleID[i]);
		  else ls.L = scene->allLights[i]->sample(dg, ls.wi, ls.tMax, sampler->getVec2f(lightSampleID));

		  /*! run through all the lighsources and compute radiance accumulatively */
		  //sum += reduce_max(scene->allLights[i]->eval(dg,ls.wi));
		  //accumRad.push_back(sum);

		  /*! Ignore zero radiance or illumination from the back. */
		  //if (ls.L == Col3f(zero) || ls.wi.pdf == 0.0f || dot(dg.Ns,Vec3f(ls.wi)) <= 0.0f) continue;
		  if (ls.L != Col3f(zero) && ls.wi.pdf != 0.0f && dot(dg.Ns,Vec3f(ls.wi)) > 0.0f) 
		  {

			  /*! Test for shadows. */
			  bool inShadow = scene->accel->occluded(Ray(dg.P, ls.wi, dg.error*epsilon, ls.tMax-dg.error*epsilon));
			  numRays++;
			  if (!inShadow) 
			  {
				  /*
				   *! Evaluate BRDF. */
// 				  L += ls.L * brdfs.eval(wo, dg, ls.wi, directLightingBRDFTypes) * rcp(ls.wi.pdf);
				  float brdfPDF = brdfs.pdf(wo, dg, ls.wi);
				  L += ls.L * brdfs.eval(wo, dg, ls.wi, directLightingBRDFTypes) * rcp(ls.wi.pdf + brdfPDF);
			  }
		  }
	  }
    }
      Sample3f wi,wii; BRDFType type;
      Vec2f s  = sampler->getVec2f(firstScatterSampleID     + lightPath.depth);
//      ss = sampler->getFloat(firstScatterTypeSampleID + lightPath.depth);
//       Col3f d = brdfs.sample(wo, dg, wii, type, s, ss, giBRDFTypes);
      
//      s  = sampler->getVec2f(firstScatterSampleID     + lightPath.depth);
     // ss = sampler->getFloat(firstScatterTypeSampleID + lightPath.depth);
      Col3f c = brdfs.sample(wo, dg, wi, type, s, ss, giBRDFTypes);
	
      
      /*VZORKUJEM BRDF*/
      
      
      
      
      
      
       //L+= sampleBRDF( scene, dg, c, wi );
      
       
       /*
       
         Col3f red = Col3f(1.0f,0.0f,0.0f);
	  Col3f blue = Col3f(0.0f,0.0f,1.0f);
	  if ( wi.pdf < 0.0f )
	  return zero;

	  Ray r( dg.P, wi.value);
	  DifferentialGeometry diff;
	  scene->accel->intersect( r, diff );
	  scene->postIntersect( r, diff );

	  Col3f radiance = Col3f( 0.0f,0.0f,0.0f );
	  if ( diff.light  )
	    radiance = diff.light->Le(diff, r.dir );

	  

	  L += radiance * c / wi.pdf;// *  dot( diff.Ng, -wi.value )) ;//  / wi.pdf);
	  
       
       
       */
       
       
       
       
       
       
       
      /* #VZORKUJEM BRDF*/
      
	/* Add the resulting light */
 	Lsum += coeff * L;

    /*! Global illumination. Pick one BRDF component and sample it. */
    if (lightPath.depth < maxDepth) //always true
    {
      /*! sample brdf */
//       Sample3f wi; BRDFType type;
//       Vec2f s  = sampler->getVec2f(firstScatterSampleID     + lightPath.depth);
//       float ss = sampler->getFloat(firstScatterTypeSampleID + lightPath.depth);
//       Col3f c = brdfs.sample(wo, dg, wi, type, s, ss, giBRDFTypes);
      
      //if (useDirectLighting)
	//L += coeff * sampleBRDF( lightPath, scene, sampler, dg, c, type, wi );
	  
      /*! Continue only if we hit something valid. */
      if (c != Col3f(zero) && wi.pdf > 0.0f)
      {
        /*! Compute  simple volumetric effect. */
        const Col3f& transmission = lightPath.lastMedium.transmission;
        if (transmission != Col3f(one)) c *= pow(transmission,dg.t);

        /*! Tracking medium if we hit a medium interface. */
        Medium nextMedium = lightPath.lastMedium;
        if (type & TRANSMISSION) nextMedium = dg.material->nextMedium(lightPath.lastMedium);

        /*! Continue the path according the prob of survival. */
		float q = 1; 

		/*! Start using the Russian Roulette after first lambertian reflection */ 
// 		if (doneDiffuse) 
		{ 
			/*! Pr(ray survival) computation - reflectance estimation */
			q = min(abs(reduce_max(c) * rcp(wi.pdf)), (float)1);
			/*! Generate the random value */
			unsigned int RndVal;
			(rand_r(&RndVal)); // std::cout << "\nRND gen error!\n";
			if ((float)RndVal/(float)UINT_MAX > q)
				return Lsum;// Ray kill => return accum value.
		}

		/*! Continue the path */
		lightPath = lightPath.extended(Ray(dg.P, wi, dg.error*epsilon, inf), nextMedium, c, (type & directLightingBRDFTypes) != NONE);
		
		/*! Accumulate the throughput coefficient */
		coeff = coeff * c * rcp(q * wi.pdf);

		/* Lambertian reflection check */
		if (wi.pdf <= 0.0) 
		  doneDiffuse = true;

      }else done = true;
    }
  }

  /*! Return the accumulated value */
	return Lsum;

  }

  Col3f PathTraceIntegrator::Li(const Ray& ray, const Ref<BackendScene>& scene, Sampler* sampler, size_t& numRays) {
    return Li(LightPath(ray),scene,sampler,numRays);
  }
}
