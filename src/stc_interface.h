
#ifndef STC_INTERFACE_H
#define STC_INTERFACE_H


#ifdef _WIN32
#define LIBRARY_API extern "C" __declspec(dllexport)
#else
#define LIBRARY_API extern "C"
#endif


LIBRARY_API int stc_hide(uint cover_length, int* cover, float* costs,
             uint message_length, u8* message, int* stego);

LIBRARY_API int stc_unhide(uint stego_length, int* stego, 
               uint message_length, u8* message);

#endif
