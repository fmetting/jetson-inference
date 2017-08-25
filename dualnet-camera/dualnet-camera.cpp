/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "gstCamera.h"

#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>
//#include <conio.h>
//#include <iostream.h>

#include "cudaMappedMemory.h"
#include "cudaNormalize.h"
#include "cudaFont.h"
#include "loadImage.h"
#include "detectNet.h"
#include "imageNet.h"

#define DEFAULT_CAMERA 1	// -1 for onboard camera, or change to index of /dev/video V4L2 camera (>=0)	
		

bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}

/*
int kbhit()
{
        struct timeval tv;
        fd_set fds;
        tv.tv_sec = 0;
        tv.tv_usec = 0;
        FD_ZERO(&fds);
        FD_SET(STDIN_FILENO, &fds);
        select (STDIN_FILENO+1, &fds, NULL, NULL, &tv);
        return FD_ISSET(STDIN_FILENO, &fds);
}
*/

int main( int argc, char** argv )
{
	printf("dualnet-camera (FM fork)\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	

	/*
	 * parse network type from CLI arguments
	 */
	detectNet::NetworkType networkType = detectNet::CARDNET;
        imageNet::NetworkType img_networkType = imageNet::ALEXNET_54CARDS;

	if( argc > 1 )
	{
		if( strcmp(argv[1], "multiped") == 0 )
			networkType = detectNet::PEDNET_MULTI;
		else if( strcmp(argv[1], "cardnet") == 0 )
			networkType = detectNet::CARDNET;
                else if( strcmp(argv[1], "ped-100") == 0 )
			networkType = detectNet::PEDNET;
		else if( strcmp(argv[1], "facenet") == 0 )
			networkType = detectNet::FACENET;
                else networkType = detectNet::CARDNET;
	}
	
	
	if( argc > 2 )
	{
		if( strcmp(argv[2], "alexnet") == 0 )
			img_networkType = imageNet::ALEXNET;
		else if( strcmp(argv[2], "googlenet") == 0 )
			img_networkType = imageNet::GOOGLENET;
                else if( strcmp(argv[2], "alexnet_54cards") == 0 )
			img_networkType = imageNet::ALEXNET_54CARDS;
                else img_networkType = imageNet::ALEXNET_54CARDS;
	}
	


         if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	/*
	 * create the camera device
	 */
	gstCamera* camera = gstCamera::Create(DEFAULT_CAMERA);
	
	if( !camera )
	{
		printf("\ndualnet-camera:  failed to initialize video device\n");
		return 0;
	}
	
	printf("\ndualnet-camera:  successfully initialized video device\n");
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
	

	/*
	 * create detectNet
	 */

        
	detectNet* net = detectNet::Create(networkType);
	
	if( !net )
	{
		printf("detectnet: failed to initialize imageNet\n");
		return 0;
	}

         /*
	 * create imageNet
	 */

       // imageNet::NetworkType networkType = imageNet::ALEXNET;
	imageNet* net2 = imageNet::Create(img_networkType);

	if( !net2 )
	{
		printf("imagenet: failed to initialize imageNet\n");
		return 0;
	}


	/*
	 * allocate memory for output bounding boxes and class confidence
	 */
	const uint32_t maxBoxes = net->GetMaxBoundingBoxes();		printf("maximum bounding boxes:  %u\n", maxBoxes);
	const uint32_t classes  = net->GetNumClasses();
	
	float* bbCPU    = NULL;
	float* bbCUDA   = NULL;
	float* confCPU  = NULL;
	float* confCUDA = NULL;
	
	if( !cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
	    !cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)) )
	{
		printf("dualnet-console:  failed to alloc output memory\n");
		return 0;
	}
	

	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	glTexture* texture = NULL;
	
	if( !display ) {
		printf("\ndualnet-camera:  failed to create openGL display\n");
	}
	else
	{
		texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);

		if( !texture )
			printf("dualnet-camera:  failed to create openGL texture\n");
	}
	
	
	/*
	 * create font
	 */
	cudaFont* font = cudaFont::Create();
	

	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("\ndualnet-camera:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("\ndualnet-camera:  camera open for streaming\n");
	
	/*
	 * processing loop
	 */
	float confidence = 0.0f;
	int    imgWidth  = 1280;
	int    imgHeight = 720;	
        char c;
        FILE *fp_out;
        int bbox_x1[maxBoxes] = { 0 };
        int bbox_y1[maxBoxes] = { 0 };
        int bbox_x2[maxBoxes] = { 0 };
        int bbox_y2[maxBoxes] = { 0 };
        int crop_width = 0;
        int crop_height = 0;
        int filecount = 0;
        int cropcount = 0;
        int boxcount = 0;
        char str[256];
        
	while( !signal_recieved )
	{
		void* imgCPU  = NULL;
		void* imgCUDA = NULL;
		
		// get the latest frame
		if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
			printf("\ndualnet-camera:  failed to capture frame\n");

               /*
                if(kbhit()) {
                c=fgetc(stdin);  
                printf("Creating TXT file\n"); 
                char txtfilename[32];
                snprintf(txtfilename, sizeof(char)*32, "test/DTN_%i.txt", filecount);
                char pngfilename[32];
                snprintf(pngfilename, sizeof(char)*32, "test/DTN_%i.png", filecount);
                char cropfilename[32];

                // save image to disk 
	        printf("dualnet-camera:  writing %ix%i image to '%s'\n", imgWidth, imgHeight, pngfilename);
		if( !saveImageRGB(pngfilename, (uchar3*)imgCPU, imgWidth, imgHeight) )
		   printf("dualnet-camera:  failed saving %ix%i image to '%s'\n", imgWidth, imgHeight, pngfilename);
		 else	
		  printf("dualnet-camera:  successfully wrote %ix%i image to '%s'\n", imgWidth, imgHeight, pngfilename);

                if (boxcount > 0)
                {
                   
                   fp_out = fopen(txtfilename, "w");
                   
                   for( int n1=0; n1 < boxcount; n1++ ) {
                
                   if (fp_out != NULL) {
                   fprintf(fp_out, "PlayingCard 0.0 0 0.0 %u %u %u %u 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n", bbox_x1[n1], bbox_y1[n1], bbox_x2[n1], bbox_y2[n1]);
                   } else
                   printf("The TXT file is not opened\n");

                 // save crop to disk
                 snprintf(cropfilename, sizeof(char)*32, "test/CROP%i_%i.png", cropcount, filecount); 
	         if( !saveImgCropRGB(cropfilename, (uchar3*)imgCPU, bbox_x1[n1], bbox_y1[n1], bbox_x2[n1], bbox_y2[n1]) )
		   printf("dualnet-crop:  failed saving %ix%i image to '%s'\n", (bbox_x2[n1]-bbox_x1[n1]), (bbox_y2[n1]-bbox_y1[n1]), cropfilename);
		 else	
		  printf("dualnet-crop:  successfully wrote %ix%i image to '%s'\n", (bbox_x2[n1]-bbox_x1[n1]), (bbox_y2[n1]-bbox_y1[n1]), cropfilename);

                  cropcount++;
                 } // end for
                  fclose(fp_out); 
                 } // end boxcount if
                   cropcount = 0;
                   filecount++;
                 } // end kbhit if

*/

		// convert from YUV to RGBA
		void* imgRGBA = NULL;
		
		if( !camera->ConvertRGBA(imgCUDA, &imgRGBA) )
		printf("dualnet-camera:  failed to convert from NV12 to RGBA\n");


// FM Note: Use this #if to comment out the nnet stuff and just get live video.
#if 1

		// classify image with detectNet
		int numBoundingBoxes = maxBoxes;


		if( net->Detect((float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), bbCPU, &numBoundingBoxes, confCPU))
		{
			printf("%i bounding boxes detected\n", numBoundingBoxes);
		
			int lastClass = 0;
			int lastStart = 0;
			boxcount = numBoundingBoxes;

			for( int n=0; n < numBoundingBoxes; n++ )
			{
				const int nc = confCPU[n*2+1];
				float* bb = bbCPU + (n * 4);
				
				printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]); 
                                
                                if (bb[0] < 0.0) bbox_x1[n] = 0; 
                                 else bbox_x1[n] = (int) (bb[0]+.5);
                                if (bb[1] < 0.0) bbox_y1[n] = 0;
                                 else bbox_y1[n] = (int) (bb[1]+.5);
                                if (bb[2] > 1279) bbox_x2[n] = 1279;
                                 else bbox_x2[n] = (int) (bb[2]+.5);
                                if (bb[3] > 719) bbox_y2[n] = 719;
                                 else bbox_y2[n] = (int) (bb[3]+.5);

                                crop_width = bbox_x2[n] - bbox_x1[n]; 
                                crop_height = bbox_y2[n] - bbox_y1[n];
                                const int img_class = net2->ClassifyROI((float*)imgRGBA, crop_width, crop_height, bbox_x1[n], bbox_y1[n], &confidence);
		                if( img_class >= 0 )
		                {
			          printf("imagenet-camera:  %2.5f%% class #%i (%s)\n", confidence * 100.0f, img_class, net2->GetClassDesc(img_class)); 
                                }

                               if( font != NULL ) {
				
				sprintf(str, "%s", net2->GetClassDesc(img_class));
				
				font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
								    str, (bbox_x1[n]+10), ((bbox_y2[n]-bbox_y1[n])/2+bbox_y1[n]-10), make_float4(255.0f, 255.0f, 0.0f, 255.0f));
			        }	

//--------------------------------------------------------------------------
				
                       
				if( nc != lastClass || n == (numBoundingBoxes - 1) )
				{
					if( !net->DrawBoxes((float*)imgRGBA, (float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), 
						                        bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
						printf("detectnet:  failed to draw boxes\n");
						
					lastClass = nc;
					lastStart = n;

					CUDA(cudaDeviceSynchronize());
				}
			}
		
			/*if( font != NULL )
			{
				char str[256];
				sprintf(str, "%05.2f%% %s", confidence * 100.0f, net->GetClassDesc(img_class));
				
				font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
								    str, 10, 10, make_float4(255.0f, 255.0f, 255.0f, 255.0f));
			}*/
			
			if( display != NULL )
			{
				char str[256];
				sprintf(str, "TensorRT build %x | %s | %04.1f FPS", NV_GIE_VERSION, net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
				//sprintf(str, "GIE build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
				display->SetTitle(str);	
			}
		}
#else

		/*if( display != NULL )
		{
			printf("FPS: %04.1f\n", display->GetFPS());
		}*/

#endif


		// update display
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();

			if( texture != NULL )
			{
				// rescale image pixel intensities for display
				CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f), 
								   (float4*)imgRGBA, make_float2(0.0f, 1.0f), 
		 						   camera->GetWidth(), camera->GetHeight()));

				// map from CUDA to openGL using GL interop
				void* tex_map = texture->MapCUDA();

				if( tex_map != NULL )
				{
					cudaMemcpy(tex_map, imgRGBA, texture->GetSize(), cudaMemcpyDeviceToDevice);
					texture->Unmap();
				}

				// draw the texture
				texture->Render(10,10);		
			}

			display->EndRender();
		}
	}
	
	printf("\ndualnet-camera:  un-initializing video device\n");
	
	
	/*
	 * shutdown the camera device
	 */
	if( camera != NULL )
	{
		delete camera;
		camera = NULL;
	}

	if( display != NULL )
	{
		delete display;
		display = NULL;
	}
	
	printf("dualnet-camera:  video device has been un-initialized.\n");
	printf("dualnet-camera:  this concludes the test of the video device.\n");
	return 0;
}

