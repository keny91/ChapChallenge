


/**	NOTE ***
 * 		
 * 
 * 
 * */






//-----------------------------------------------------------------------------------------------------------------------
//
//Ejemplo de código que inicializa el buffer de imágenes, así como los arrays de punteros a los perfiles de las imágenes.
//
//-----------------------------------------------------------------------------------------------------------------------

/**
 * 	In a few words, we will be initializing an 'countOfImageBuffers' number of buffers.
 * 


 * 	unsigned short is the right value to store pixel representation, matching the 16bit depth I worked with in my challenges.
 * 
 * */
void InitializeImagesBuffers(int countOfImageBuffers)
{
	unsigned short		**_ppBufferImages ; //Pointers to image buffers
	unsigned short		***_pppLinesBufferImages ;//Pointers to profiles of image buffers


	//Initialize buffers of images and array of pointers
	//to each profile of each buffer
	_ppBufferImages = new unsigned short*[countOfImageBuffers] ;  			// 	pointer to image buffer
	_pppLinesBufferImages = new unsigned short **[countOfImageBuffers];		//	pointer to an array of values tracking bit memory of images

	// we will now initialize 
	for( int idx = 0;  idx < countOfImageBuffers; idx++ )
	{
		//_deviceWidth = 2048
		//_deviceWidth = 2000

		_ppBufferImages[idx] = new unsigned short[_deviceWidth*_deviceHeight] ;  //  alloc the matrix size
		memset(_ppBufferImages[idx], 0, sizeof(unsigned short)*_deviceWidth*_deviceHeight ) ;  // init mat as 0s

		_pppLinesBufferImages[idx] = new unsigned short*[_deviceHeiofght];  // we prepare the space to store image rows
		for(int jdx = 0; jdx < _deviceHeight; jdx++) 
		{
			_pppLinesBufferImages[idx][jdx] = 					// in each row/entry we 
				_ppBufferImages[idx] + _deviceWidth*jdx  ;		// each entry should contain increments of _deviceWidth
		}														// from 0 to _deviceWidth*_deviceWidth
	}
}

//--------------------------------------------------------------------------------------------------------
//
//Ejemplo de código con estructuras anidadas, acceso directo a memoria, gestión de excepciones y semáforos. 
//
//--------------------------------------------------------------------------------------------------------
void AddImage( const int& palletNumber, 
const unsigned char** const ppBuffer	// we are not allowed to modify the buffer here
	const DeviceConfiguration* deviceConfig, 
	const FileFormat &fileFormat,
	double vMarkFirstProfile )
{
	int			slot, prevPalletNumber ;				// 
	CString deviceName = deviceConfig->GetName() ;		// camera name 
	ReposPalletsMap::const_iterator iPalletNumber ;		//	

	//Ejemplos de estructuras de datos
	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	//struct ImageData
	//{
	//	bool						emptySlot ;
	//	unsigned char		*imageBuffer;
	//	unsigned long		allocSize;  //Allocated size.
	//	int							ownerPallet;
	//};
	vector< ImageData > _imageMemory;

	//struct PalletData
	//{
	//	int     imageData ; //Location in ImageData vector.
	//	bool    bufferFull ;
	//	VARIANT drawSegmentation ;
	//	VARIANT drawBinarization ;
	//	VARIANT drawDefectCandidate ;
	//	const DeviceConfiguration* deviceConfig ;
	//	FileFormat fileFormat ;
	//} ;
	//typedef map<CString, PalletData> PalletDataMap ;
	//typedef map<int, PalletDataMap> ReposPalletsMap ;
	ReposPalletsMap _pallets;
	//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



	try
	{
		//Entering Critical Section.
		//-------------------------------------------------------------
		WaitForSingleObject( _mapmutex, INFINITE ); // lock mutex while we work in this process
													// make sure no other threads tries to write at the same tim
													// wait until is unlocked
		//Looking for the next available memory slot for the new image.
		
		slot = _nextMemorySlot ;
		_nextMemorySlot = (_nextMemorySlot + 1) % _maxMemorySlots ;  // this way we will cycle yo the first position
																	// once we have passed max memory slots 

		//If the slot to be used is currently used by another pallet
		//then this old pallet has to be removed from memory now.
		//_imageMemory es de tipo vector<ImageData>

		// I find it strange that we just clear existing entries without having a way to check
		// if we are done processing them. I would assume is done here
		if ( !_imageMemory[slot].emptySlot )
		{
			prevPalletNumber = _imageMemory[slot].ownerPallet ;		// set the identifier of the previous pallet done in this slot
			RemovePalletInformation( prevPalletNumber );			// clear
			_pallets.erase( prevPalletNumber );
			_classifications.erase( prevPalletNumber );

			//Mark as empty each slot with the same pallet number
			//and compact the iamges buffer
			RemovePalletFromImagesBuffer( prevPalletNumber, slot ) ;
		}

		//Mark the slot as unavaliable
		_imageMemory[slot].emptySlot = false ;

		//In order to save in memory the image, it can be reused the memory
		//previously allocated in the slot. This will be possible only if the 
		//old image's size is the same that the new one (usually true).
		int lineSize = deviceConfig->GetWidth() * deviceConfig->GetBytes();  // based on camera specs
		int height = deviceConfig->GetHeight();
		unsigned long newSize = height * lineSize;

		// maybe the previous camera had different specs, in any case
		if ( _imageMemory[slot].allocSize >= newSize )
		{
			//Copying the buffer in the previously allocated memory.
			for (int i=0; i < height; i++)
			{
				memcpy( _imageMemory[slot].imageBuffer + i*(size_t)lineSize, ppBuffer[i], 	// we will copy line by line
					lineSize );																
			}
			_imageMemory[slot].ownerPallet = palletNumber;		// set new entry

		}
		else //The image has a bigger size from previous (or it's the 1st).
		{
			// and not null buffer then we clear it
			if ( _imageMemory[slot].imageBuffer ) 
			{
				delete [] _imageMemory[slot].imageBuffer;
			}

			//	if  
			unsigned char* newBuffer = new unsigned char[newSize];
			//  if couldnt allocate the memory
			if ( !newBuffer ) 
			{
				ReleaseMutex( _mapmutex );

				Utilities::ThrowException("ResultRepository::AddImage",
					ID_EXCEPT_RESULT_REPOSITORY_OUTOFMEMORY);
			}


			for (int i=0; i < height; i++)
			{
				memcpy( newBuffer + i*(size_t)lineSize, ppBuffer[i],
					lineSize );
			}

			// set new
			_imageMemory[slot].imageBuffer = newBuffer;
			_imageMemory[slot].allocSize = newSize;
			_imageMemory[slot].ownerPallet = palletNumber;
		}

		//Connecting the memory slot to its (pallet/device) owner.
		_pallets[palletNumber][deviceName].imageData = slot;
		_pallets[palletNumber][deviceName].bufferFull = false;
		_pallets[palletNumber][deviceName].fileFormat = fileFormat;

		//Storing the device configuration.
		_pallets[palletNumber][deviceName].deviceConfig = deviceConfig;

		//Storing it in permanent repository.
		SavePalletInformation( palletNumber, deviceConfig, fileFormat, vMarkFirstProfile );

		//-------------------------------------------------------------
		ReleaseMutex( _mapmutex ); // mutex released
	}

	// if an error happens, report the error, set where in the execution happens
	// and 	
	catch(Exception *e )
	{
		CString function = e->GetFunction() ;
		CString msg = e->GetAuxiliarMsg() ;

		function += " <- ResultRepository::AddImage" ;
		msg.Format( "%s <- Execution point #%d", msg.GetString(), executionpoint ) ;
		Utilities::ThrowException(function,ID_EXCEPT_RESULT_REPOSITORY,msg);
	}
	// catch other exeptons?
	catch(...)
	{
		CString msg;

		ReleaseMutex( _mapmutex ); // release

		msg.Format("Fail at execution point #%d", executionpoint);
		Utilities::ThrowException("ResultRepository::AddImage",
			ID_EXCEPT_RESULT_REPOSITORY, msg);
	}
}
