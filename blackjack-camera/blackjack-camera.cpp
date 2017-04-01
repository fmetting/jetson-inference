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

#define DEFAULT_CAMERA -1	// -1 for onboard camera, or change to index of /dev/video V4L2 camera (>=0)	
		

bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}


int main( int argc, char** argv )
{
	printf("detectnet-camera\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	/*
	 * create the camera device
	 */
	gstCamera* camera = gstCamera::Create(DEFAULT_CAMERA);
	
	if( !camera )
	{
		printf("\ndetectnet-camera:  failed to initialize video device\n");
		return 0;
	}
	
	printf("\ndetectnet-camera:  successfully initialized video device\n");
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
	

	/*
	 * create detectNet
	 */

        detectNet::NetworkType networkType = detectNet::CARDNET;
	detectNet* net = detectNet::Create(networkType);
	
	if( !net )
	{
		printf("detectnet:   failed to initialize imageNet\n");
		return 0;
	}

         /*
	 * create imageNet
	 */

        imageNet::NetworkType img_networkType = imageNet::ALEXNET_54CARDS;
	imageNet* net2 = imageNet::Create(img_networkType);

	if( !net2 )
	{
		printf("imagenet:   failed to initialize imageNet\n");
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
		printf("detectnet-console:  failed to alloc output memory\n");
		return 0;
	}
	

	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	glTexture* texture = NULL;
	
	if( !display ) {
		printf("\nblackjack-camera:  failed to create openGL display\n");
	}
	else
	{
		texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);

		if( !texture )
			printf("blackjack-camera:  failed to create openGL texture\n");
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
		printf("\nblackjack-camera:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("\nblackjack-camera:  camera open for streaming\n");
	

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
        char str[256];
        int player_value = 0;
        int computer_value = 0;
        int player_stand = 0;

	while( !signal_recieved )
	{
		void* imgCPU  = NULL;
		void* imgCUDA = NULL;
		
		// get the latest frame
		if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
			printf("\nblackjack-camera:  failed to capture frame\n");

                // convert from YUV to RGBA
		void* imgRGBA = NULL;
		
		if( !camera->ConvertRGBA(imgCUDA, &imgRGBA) )
		printf("blackjack-camera:  failed to convert from NV12 to RGBA\n");

 
		// classify image with detectNet
		int numBoundingBoxes = maxBoxes;
                   
                player_value = 0;
                computer_value = 0;
                player_stand = 0;
 
		if( net->Detect((float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), bbCPU, &numBoundingBoxes, confCPU))
		{
			printf("%i bounding boxes detected\n", numBoundingBoxes);
		
			int lastClass = 0;
			int lastStart = 0;

			for( int n=0; n < numBoundingBoxes; n++ )
			{
				const int nc = confCPU[n*2+1];
				float* bb = bbCPU + (n * 4);
				
				printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]); 
                                bbox_x1[n] = (int) (bb[0]+.5);
                                bbox_y1[n] = (int) (bb[1]+.5);
                                bbox_x2[n] = (int) (bb[2]+.5);
                                bbox_y2[n] = (int) (bb[3]+.5);
                                crop_width = bbox_x2[n] - bbox_x1[n]; 
                                crop_height = bbox_y2[n] - bbox_y1[n];
                                const int img_class = net2->ClassifyROI((float*)imgRGBA, crop_width, crop_height, bbox_x1[n], bbox_y1[n], &confidence);
		                if( img_class >= 0 )
                                {
			          printf("imagenet:  %2.5f%% class #%i (%s)\n", confidence * 100.0f, img_class, net2->GetClassDesc(img_class));
                                  if (bbox_y1[n] < 360) {
                                   switch (img_class) {
                                    case 0 : player_value = player_value + 10; //10ofClubs
                                             break;
                                    case 1 : player_value = player_value + 10; //10ofDiamonds
                                             break;
                                    case 2 : player_value = player_value + 10; //10ofHearts
                                             break;
                                    case 3 : player_value = player_value + 10; //10ofSpades
                                             break;
                                    case 4 : player_value = player_value + 2; //2ofClubs
                                             break;
                                    case 5 : player_value = player_value + 2; //2ofDiamonds
                                             break;
                                    case 6 : player_value = player_value + 2; //2ofHearts
                                             break;
                                    case 7 : player_value = player_value + 2; //2ofSpades
                                             break;
                                    case 8 : player_value = player_value + 3; //3ofClubs
                                             break;
                                    case 9 : player_value = player_value + 3; //3ofDiamonds
                                             break;
                                    case 10 : player_value = player_value + 3; //3ofHearts
                                             break;
                                    case 11 : player_value = player_value + 3; //3ofSpades
                                             break;
                                    case 12 : player_value = player_value + 4; //4ofClubs
                                             break;
                                    case 13 : player_value = player_value + 4; //4ofDiamonds
                                             break;
                                    case 14 : player_value = player_value + 4; //4ofHearts
                                             break;
                                    case 15 : player_value = player_value + 4; //4ofSpades
                                             break;
                                    case 16 : player_value = player_value + 5; //5ofClubs
                                             break;
                                    case 17 : player_value = player_value + 5; //5ofDiamonds
                                             break;
                                    case 18 : player_value = player_value + 5; //5ofHearts
                                             break;
                                    case 19 : player_value = player_value + 5; //5ofSpades
                                             break;
                                    case 20 : player_value = player_value + 6; //6ofClubs
                                             break;
                                    case 21 : player_value = player_value + 6; //6ofDiamonds
                                             break;
                                    case 22 : player_value = player_value + 6; //6ofHearts
                                             break;
                                    case 23 : player_value = player_value + 6; //6ofSpades
                                             break;
                                    case 24 : player_value = player_value + 7; //7ofClubs
                                             break;
                                    case 25 : player_value = player_value + 7; //7ofDiamonds
                                             break;
                                    case 26 : player_value = player_value + 7; //7ofHearts
                                             break;
                                    case 27 : player_value = player_value + 7; //7ofSpades
                                             break;
                                    case 28 : player_value = player_value + 8; //8ofClubs
                                             break;
                                    case 29 : player_value = player_value + 8; //8ofDiamonds
                                             break;
                                    case 30 : player_value = player_value + 8; //8ofHearts
                                             break;
                                    case 31 : player_value = player_value + 8; //8ofSpades
                                             break;
                                    case 32 : player_value = player_value + 9; //9ofClubs
                                             break;
                                    case 33 : player_value = player_value + 9; //9ofDiamonds
                                             break;
                                    case 34 : player_value = player_value + 9; //9ofHearts
                                             break;
                                    case 35 : player_value = player_value + 9; //9ofSpades
                                             break;
                                    case 36 : player_value = player_value + 1; //AceofClubs
                                             break;
                                    case 37 : player_value = player_value + 1; //AceofDiamonds
                                             break;
                                    case 38 : player_value = player_value + 1; //AceofHearts
                                             break;
                                    case 39 : player_value = player_value + 1; //AceofSpades
                                             break;
                                    case 40 : player_value = player_value + 0; //BlackJoker
                                             break;
                                    case 41 : player_value = player_value + 10; //JackofClubs
                                             break;
                                    case 42 : player_value = player_value + 10; //JackofDiamonds
                                             break;
                                    case 43 : player_value = player_value + 10; //JackofHearts
                                             break;
                                    case 44 : player_value = player_value + 10; //JackofSpades
                                             break;
                                    case 45 : player_value = player_value + 10; //KingofClubs
                                             break;
                                    case 46 : player_value = player_value + 10; //KingofDiamonds
                                             break;
                                    case 47 : player_value = player_value + 10; //KingofHearts
                                             break;
                                    case 48 : player_value = player_value + 10; //KingofSpades
                                             break;
                                    case 49 : player_value = player_value + 10; //QueenofClubs
                                             break;
                                    case 50 : player_value = player_value + 10; //QueenofDiamonds
                                             break;
                                    case 51 : player_value = player_value + 10; //QueenofHearts
                                             break;
                                    case 52 : player_value = player_value + 10; //QueenofSpades
                                             break;
                                    case 53 : player_stand = 1; //RedJoker
                                             break;
                                    default : player_value = player_value + 0;
                                             break; } // end switch case
                                    } // end if bbox 
                                   else {
                                   switch (img_class) {
                                    case 0 : computer_value = computer_value + 10; //10ofClubs
                                             break;
                                    case 1 : computer_value = computer_value + 10; //10ofDiamonds
                                             break;
                                    case 2 : computer_value = computer_value + 10; //10ofHearts
                                             break;
                                    case 3 : computer_value = computer_value + 10; //10ofSpades
                                             break;
                                    case 4 : computer_value = computer_value + 2; //2ofClubs
                                             break;
                                    case 5 : computer_value = computer_value + 2; //2ofDiamonds
                                             break;
                                    case 6 : computer_value = computer_value + 2; //2ofHearts
                                             break;
                                    case 7 : computer_value = computer_value + 2; //2ofSpades
                                             break;
                                    case 8 : computer_value = computer_value + 3; //3ofClubs
                                             break;
                                    case 9 : computer_value = computer_value + 3; //3ofDiamonds
                                             break;
                                    case 10 : computer_value = computer_value + 3; //3ofHearts
                                             break;
                                    case 11 : computer_value = computer_value + 3; //3ofSpades
                                             break;
                                    case 12 : computer_value = computer_value + 4; //4ofClubs
                                             break;
                                    case 13 : computer_value = computer_value + 4; //4ofDiamonds
                                             break;
                                    case 14 : computer_value = computer_value + 4; //4ofHearts
                                             break;
                                    case 15 : computer_value = computer_value + 4; //4ofSpades
                                             break;
                                    case 16 : computer_value = computer_value + 5; //5ofClubs
                                             break;
                                    case 17 : computer_value = computer_value + 5; //5ofDiamonds
                                             break;
                                    case 18 : computer_value = computer_value + 5; //5ofHearts
                                             break;
                                    case 19 : computer_value = computer_value + 5; //5ofSpades
                                             break;
                                    case 20 : computer_value = computer_value + 6; //6ofClubs
                                             break;
                                    case 21 : computer_value = computer_value + 6; //6ofDiamonds
                                             break;
                                    case 22 : computer_value = computer_value + 6; //6ofHearts
                                             break;
                                    case 23 : computer_value = computer_value + 6; //6ofSpades
                                             break;
                                    case 24 : computer_value = computer_value + 7; //7ofClubs
                                             break;
                                    case 25 : computer_value = computer_value + 7; //7ofDiamonds
                                             break;
                                    case 26 : computer_value = computer_value + 7; //7ofHearts
                                             break;
                                    case 27 : computer_value = computer_value + 7; //7ofSpades
                                             break;
                                    case 28 : computer_value = computer_value + 8; //8ofClubs
                                             break;
                                    case 29 : computer_value = computer_value + 8; //8ofDiamonds
                                             break;
                                    case 30 : computer_value = computer_value + 8; //8ofHearts
                                             break;
                                    case 31 : computer_value = computer_value + 8; //8ofSpades
                                             break;
                                    case 32 : computer_value = computer_value + 9; //9ofClubs
                                             break;
                                    case 33 : computer_value = computer_value + 9; //9ofDiamonds
                                             break;
                                    case 34 : computer_value = computer_value + 9; //9ofHearts
                                             break;
                                    case 35 : computer_value = computer_value + 9; //9ofSpades
                                             break;
                                    case 36 : computer_value = computer_value + 1; //AceofClubs
                                             break;
                                    case 37 : computer_value = computer_value + 1; //AceofDiamonds
                                             break;
                                    case 38 : computer_value = computer_value + 1; //AceofHearts
                                             break;
                                    case 39 : computer_value = computer_value + 1; //AceofSpades
                                             break;
                                    case 40 : computer_value = computer_value + 0; //BlackJoker
                                             break;
                                    case 41 : computer_value = computer_value + 10; //JackofClubs
                                             break;
                                    case 42 : computer_value = computer_value + 10; //JackofDiamonds
                                             break;
                                    case 43 : computer_value = computer_value + 10; //JackofHearts
                                             break;
                                    case 44 : computer_value = computer_value + 10; //JackofSpades
                                             break;
                                    case 45 : computer_value = computer_value + 10; //KingofClubs
                                             break;
                                    case 46 : computer_value = computer_value + 10; //KingofDiamonds
                                             break;
                                    case 47 : computer_value = computer_value + 10; //KingofHearts
                                             break;
                                    case 48 : computer_value = computer_value + 10; //KingofSpades
                                             break;
                                    case 49 : computer_value = computer_value + 10; //QueenofClubs
                                             break;
                                    case 50 : computer_value = computer_value + 10; //QueenofDiamonds
                                             break;
                                    case 51 : computer_value = computer_value + 10; //QueenofHearts
                                             break;
                                    case 52 : computer_value = computer_value + 10; //QueenofSpades
                                             break;
                                    case 53 : computer_value = computer_value + 0; //RedJoker
                                             break;
                                    default : computer_value = computer_value + 0;
                                             break; } // end switch case
                                      }// end else bbox
                                 } // end if img_class
                                

                               if( font != NULL ) {
				
				sprintf(str, "%s", net2->GetClassDesc(img_class));
				
				font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
								    str, (bbox_x1[n]+10), ((bbox_y2[n]-bbox_y1[n])/2+bbox_y1[n]-10), make_float4(255.0f, 255.0f, 0.0f, 255.0f));
			        }	


				
                               if( nc != lastClass || n == (numBoundingBoxes - 1) )
				{
					if( !net->DrawBoxes((float*)imgRGBA, (float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), 
						                        bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
						printf("detectnet-console:  failed to draw boxes\n");
						
					lastClass = nc;
					lastStart = n;

					CUDA(cudaDeviceSynchronize());
				}
			
                       } // end for


//-------------------------------------------------------------------------------
		
			
                        if( font != NULL )
			{
				if (computer_value > 21) sprintf(str, "Busted");
                                else sprintf(str, "Computer = %i", computer_value);
				font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
								    str, 10, 690, make_float4(255.0f, 255.0f, 255.0f, 255.0f));
			}

                        CUDA(cudaDeviceSynchronize());
 
                        if( font != NULL )
			{
				if (player_value > 21) sprintf(str, "Busted");
                                else sprintf(str, "Player = %i", player_value);
                                font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
								    str, 10, 5, make_float4(255.0f, 255.0f, 255.0f, 255.0f));
                        }


                       CUDA(cudaDeviceSynchronize());

                        if ((player_value > 21) && (computer_value > 21))
                          sprintf(str, "We're both losers");
                        else if ((player_value > 21) && (computer_value < 22))
                          sprintf(str, "Haha, I won");
                        else if ((player_value < 22) && (computer_value > 21))
                          sprintf(str, "Oh, Crap!!");
                        else if ((computer_value > 16) && (player_stand == 1) && (computer_value > player_value))
                          sprintf(str, "Victory is mine!");
                        else if ((computer_value > 16) && (player_stand == 1) && (player_value > computer_value))
                          sprintf(str, "Ugh, I lost");
                        else if ((computer_value > 16) && (player_stand == 1) && (player_value == computer_value))
                          sprintf(str, "No ties.. Hit");
                        else if (computer_value < 17)
                          sprintf(str, "Hit");
                        else 
                          sprintf(str, "Stand");

 
                        if( font != NULL )
			    font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
								    str, 850, 690, make_float4(255.0f, 255.0f, 255.0f, 255.0f));
			   
                        CUDA(cudaDeviceSynchronize());
                        
			
			if( display != NULL )
			{
				char str[256];
				sprintf(str, "TensorRT build %x | %s | %04.1f FPS", NV_GIE_VERSION, net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
				display->SetTitle(str);	
			}	
		}	



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
				texture->Render(100,100);		
			}

			display->EndRender();
		}
	}
	
	printf("\nblackjack-camera:  un-initializing video device\n");
	
	
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
	
	printf("blackjack-camera:  video device has been un-initialized.\n");
	printf("blackjack-camera:  this concludes the test of the video device.\n");
	return 0;
}

