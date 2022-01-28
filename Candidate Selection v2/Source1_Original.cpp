//-----------------------------------------------------------------------------------------------------------------------
//
//Ejemplo de código que inicializa el buffer de imágenes, así como los arrays de punteros a los perfiles de las imágenes.
//
//-----------------------------------------------------------------------------------------------------------------------
void InitializeImagesBuffers(int countOfImageBuffers)
{
	unsigned short		**_ppBufferImages ; //Pointers to image buffers
	unsigned short		***_pppLinesBufferImages ;//Pointers to profiles of image buffers


	//Initialize buffers of images and array of pointers
	//to each profile of each buffer
	_ppBufferImages = new unsigned short*[countOfImageBuffers] ;
	_pppLinesBufferImages = new unsigned short **[countOfImageBuffers];
	for( int idx = 0;  idx < countOfImageBuffers; idx++ )
	{
		//_deviceWidth = 2048
		//_deviceWidth = 2000

		_ppBufferImages[idx] = new unsigned short[_deviceWidth*_deviceHeight] ;
		memset(_ppBufferImages[idx], 0, sizeof(unsigned short)*_deviceWidth*_deviceHeight ) ;

		_pppLinesBufferImages[idx] = new unsigned short*[_deviceHeight];
		for(int jdx = 0; jdx < _deviceHeight; jdx++) 
		{
			_pppLinesBufferImages[idx][jdx] = 
				_ppBufferImages[idx] + _deviceWidth*jdx  ;
		}
	}
}

//--------------------------------------------------------------------------------------------------------
//
//Ejemplo de código con estructuras anidadas, acceso directo a memoria, gestión de excepciones y semáforos. 
//
//--------------------------------------------------------------------------------------------------------
void AddImage( const int& palletNumber, 
	const unsigned char** const ppBuffer,
	const DeviceConfiguration* deviceConfig, 
	const FileFormat &fileFormat,
	double vMarkFirstProfile )
{
	int			slot, prevPalletNumber ;
	CString deviceName = deviceConfig->GetName() ;
	ReposPalletsMap::const_iterator iPalletNumber ;

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
		WaitForSingleObject( _mapmutex, INFINITE );

		//Looking for the next available memory slot for the new image.
		
		slot = _nextMemorySlot ;
		_nextMemorySlot = (_nextMemorySlot + 1) % _maxMemorySlots ;

		//If the slot to be used is currently used by another pallet
		//then this old pallet has to be removed from memory now.
		//_imageMemory es de tipo vector<ImageData>
		if ( !_imageMemory[slot].emptySlot )
		{
			prevPalletNumber = _imageMemory[slot].ownerPallet ;
			RemovePalletInformation( prevPalletNumber );
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
		int lineSize = deviceConfig->GetWidth() * deviceConfig->GetBytes();
		int height = deviceConfig->GetHeight();
		unsigned long newSize = height * lineSize;
		if ( _imageMemory[slot].allocSize >= newSize )
		{
			//Copying the buffer in the previously allocated memory.
			for (int i=0; i < height; i++)
			{
				memcpy( _imageMemory[slot].imageBuffer + i*(size_t)lineSize, ppBuffer[i], 
					lineSize );
			}
			_imageMemory[slot].ownerPallet = palletNumber;

		}
		else //The image has a bigger size from previous (or it's the 1st).
		{
			if ( _imageMemory[slot].imageBuffer ) 
			{
				delete [] _imageMemory[slot].imageBuffer;
			}

			unsigned char* newBuffer = new unsigned char[newSize];
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
		ReleaseMutex( _mapmutex );
	}
	catch(Exception *e )
	{
		CString function = e->GetFunction() ;
		CString msg = e->GetAuxiliarMsg() ;

		function += " <- ResultRepository::AddImage" ;
		msg.Format( "%s <- Execution point #%d", msg.GetString(), executionpoint ) ;
		Utilities::ThrowException(function,ID_EXCEPT_RESULT_REPOSITORY,msg);
	}
	catch(...)
	{
		CString msg;

		ReleaseMutex( _mapmutex );

		msg.Format("Fail at execution point #%d", executionpoint);
		Utilities::ThrowException("ResultRepository::AddImage",
			ID_EXCEPT_RESULT_REPOSITORY, msg);
	}
}
