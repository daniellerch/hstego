
#include <stdio.h>
#include <stdlib.h>
#include <jerror.h>
#include <jpeglib.h>
#include <jpegint.h>
#include <setjmp.h>
#include <Python.h>

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif


#ifdef _WIN32
#define LIBRARY_API __declspec(dllexport)
#else
#define LIBRARY_API 
#endif



// {{{ struct my_error_mgr
struct my_error_mgr 
{
   struct jpeg_error_mgr pub;	
   jmp_buf setjmp_buffer;	
};
typedef struct my_error_mgr * my_error_ptr;
// }}}

// {{{ my_output_message()
METHODDEF(void) my_output_message (j_common_ptr cinfo)
{
   char buffer[JMSG_LENGTH_MAX];
   (*cinfo->err->format_message) (cinfo, buffer);
   fprintf(stderr, "Error: %s\n", buffer);
}
// }}}

// {{{ my_error_exit()
METHODDEF(void) my_error_exit (j_common_ptr cinfo)
{
   char buffer[JMSG_LENGTH_MAX];
   my_error_ptr myerr = (my_error_ptr) cinfo->err;
   (*cinfo->err->format_message) (cinfo, buffer);
   fprintf(stderr, "Error: %s\n", buffer);
   longjmp(myerr->setjmp_buffer, 1);
}
// }}}

// {{{ dict_add_int()
PyObject* dict_add_int(PyObject* dict, const char *key, int value)
{
   PyObject *py_key = Py_BuildValue("s", key);
   PyObject *py_value = Py_BuildValue("i", value);
   PyDict_SetItem(dict, py_key, py_value);
   Py_DecRef(py_key);
   Py_DecRef(py_value);

   return dict;
}
// }}}

// {{{ dict_add_float()
PyObject* dict_add_float(PyObject* dict, const char *key, float value)
{
   PyObject *py_key = Py_BuildValue("s", key);
   PyObject *py_value = Py_BuildValue("f", value);
   PyDict_SetItem(dict, py_key, py_value);
   Py_DecRef(py_key);
   Py_DecRef(py_value);

   return dict;
}
// }}}

// {{{ list_append_float()
PyObject* list_append_float(PyObject* list, float value)
{
   PyObject *py_value = Py_BuildValue("f", value);
   PyList_Append(list, py_value);
   Py_DecRef(py_value);

   return list;
}
// }}}

// {{{ dict_get_int()
int dict_get_int(PyObject *dict, const char* key)
{
   PyObject *py_key = Py_BuildValue("s", key);
   PyObject* item = PyDict_GetItem(dict, py_key);
   int value = (int)PyLong_AsLong(item);
   Py_DecRef(py_key);

   return value;
}
// }}}

// {{{ dict_get_object()
PyObject *dict_get_object(PyObject *dict, const char* key)
{
   PyObject *py_key = Py_BuildValue("s", key);
   PyObject* item = PyDict_GetItem(dict, py_key);
   Py_DecRef(py_key);

   return item;
}
// }}}


// {{{ read_file()
LIBRARY_API PyObject* read_file(const char *path)
{
   PyGILState_STATE gstate = PyGILState_Ensure();

   PyObject *result = PyDict_New();
   assert(PyDict_Check(result));

   struct jpeg_decompress_struct cinfo;
   struct my_error_mgr jerr;
   FILE *f = NULL;
   PyObject *py_key = NULL;
   PyObject *py_value = NULL;
   PyObject *row = NULL;
   PyObject *col = NULL;
   PyObject *dict = NULL;

   if((f = fopen(path, "rb")) == NULL)
   {
      fprintf(stderr, "Can not open file: %s\n", path);
      PyGILState_Release(gstate);
      return result;
   }

   /* set up the normal JPEG error routines, then override error_exit. */
   cinfo.err = jpeg_std_error(&jerr.pub);
   jerr.pub.error_exit = my_error_exit;
   jerr.pub.output_message = my_output_message;

   /* establish the setjmp return context for my_error_exit to use. */
   if (setjmp(jerr.setjmp_buffer)) {
      jpeg_destroy_decompress(&cinfo);
      fclose(f);
      fprintf(stderr, "Error reading file: %s\n", path);
      PyGILState_Release(gstate);
      return result;
   }

   /* initialize JPEG decompression object */
   jpeg_create_decompress(&cinfo);
   jpeg_stdio_src(&cinfo, f);

   /* save contents of markers */
   jpeg_save_markers(&cinfo, JPEG_COM, 0xFFFF);

  /* read header and coefficients */
  jpeg_read_header(&cinfo, TRUE);

  /* Set out_color_components */
   switch (cinfo.out_color_space) {
      case JCS_GRAYSCALE:
         cinfo.out_color_components = 1;
         break;
      case JCS_RGB:
         cinfo.out_color_components = 3;
         break;
      case JCS_YCbCr:
         cinfo.out_color_components = 3;
         break;
      case JCS_CMYK:
         cinfo.out_color_components = 4;
         break;
      case JCS_YCCK:
         cinfo.out_color_components = 4;
         break;
      default:
         fprintf(stderr, "Unknown color space: %d\n", cinfo.out_color_space);
         PyGILState_Release(gstate);
         return result;
   }


   // {{{ Header info 
   result = dict_add_int(result, "image_width", cinfo.image_width);
   result = dict_add_int(result, "image_height", cinfo.image_height);
   result = dict_add_int(result, "image_color_space", cinfo.out_color_space);
   result = dict_add_int(result, "image_components", cinfo.out_color_components);
   result = dict_add_int(result, "jpeg_color_space", cinfo.jpeg_color_space);
   result = dict_add_int(result, "jpeg_components", cinfo.num_components);
   result = dict_add_int(result, "progressive_mode", cinfo.progressive_mode);
   result = dict_add_int(result, "X_density", cinfo.X_density);
   result = dict_add_int(result, "Y_density", cinfo.Y_density);
   result = dict_add_int(result, "density_unit", cinfo.density_unit);
   result = dict_add_int(result, "block_size", cinfo.block_size);
   result = dict_add_int(result, "min_DCT_h_scaled_size", cinfo.min_DCT_h_scaled_size);
   result = dict_add_int(result, "min_DCT_v_scaled_size", cinfo.min_DCT_v_scaled_size);
   // }}}

   // {{{ Components info
   PyObject* comp_info = PyList_New(0);
   assert(PyList_Check(comp_info));

   for(int ci = 0; ci < cinfo.num_components; ci++) 
   {
      PyObject *comp = PyDict_New();
      assert(PyDict_Check(comp));

      comp = dict_add_int(comp, "component_id", cinfo.comp_info[ci].component_id);
      comp = dict_add_int(comp, "h_samp_factor", cinfo.comp_info[ci].h_samp_factor);
      comp = dict_add_int(comp, "v_samp_factor", cinfo.comp_info[ci].v_samp_factor);
      comp = dict_add_int(comp, "quant_tbl_no", cinfo.comp_info[ci].quant_tbl_no);
      comp = dict_add_int(comp, "ac_tbl_no", cinfo.comp_info[ci].ac_tbl_no);
      comp = dict_add_int(comp, "dc_tbl_no", cinfo.comp_info[ci].dc_tbl_no);

      jpeg_component_info *compptr = cinfo.comp_info + ci;
      comp = dict_add_int(comp, "DCT_h_scaled_size", compptr->DCT_h_scaled_size);
      comp = dict_add_int(comp, "DCT_v_scaled_size", compptr->DCT_v_scaled_size);

      PyList_Append(comp_info, comp);
      Py_DecRef(comp);
   }

   py_key = Py_BuildValue("s", "comp_info");
   PyDict_SetItem(result, py_key, comp_info);
   Py_DecRef(py_key);

   Py_DecRef(comp_info);
   // }}}

   // {{{ Quantization tables 
   PyObject* quant_tables = PyList_New(0);
   assert(PyList_Check(quant_tables));

   for(size_t n = 0; n < NUM_QUANT_TBLS; n++) 
   {
      if(cinfo.quant_tbl_ptrs[n] != NULL) 
      {
         JQUANT_TBL *quant_ptr = cinfo.quant_tbl_ptrs[n];
         
         row = PyList_New(0);
         assert(PyList_Check(row));
         for(size_t i = 0; i < DCTSIZE; i++) 
         {
            col = PyList_New(0);
            assert(PyList_Check(col));
            for(size_t j = 0; j < DCTSIZE; j++)
            {
               py_value = Py_BuildValue("f", (double) quant_ptr->quantval[i*DCTSIZE+j]);
               PyList_Append(col, py_value);
               Py_DecRef(py_value);
            }
            PyList_Append(row, col);
            Py_DecRef(col);
         }
         PyList_Append(quant_tables, row);
         Py_DecRef(row);
         
      }
   }

   py_key = Py_BuildValue("s", "quant_tables");
   PyDict_SetItem(result, py_key, quant_tables);
   Py_DecRef(quant_tables);
   Py_DecRef(py_key);
   // }}}

   // {{{ AC Huffman tables 
   PyObject* ac_huff_tables = PyList_New(0);
   assert(PyList_Check(ac_huff_tables));

   for(size_t n = 0; n < NUM_HUFF_TBLS; n++) 
   {
      if(cinfo.ac_huff_tbl_ptrs[n] != NULL) 
      {
         JHUFF_TBL *huff_ptr = cinfo.ac_huff_tbl_ptrs[n];

         dict = PyDict_New();
         assert(PyDict_Check(dict));

         row = PyList_New(0);
         assert(PyList_Check(row));
         for(size_t i=1; i<=16; i++)
            list_append_float(row, huff_ptr->bits[i]);
         
         py_key = Py_BuildValue("s", "counts");
         PyDict_SetItem(dict, py_key, row);
         Py_DecRef(py_key);
         Py_DecRef(row);

         row = PyList_New(0);
         assert(PyList_Check(row));
         for(size_t i=0; i<256; i++)
            list_append_float(row, huff_ptr->huffval[i]);
         
         py_key = Py_BuildValue("s", "symbols");
         PyDict_SetItem(dict, py_key, row);
         Py_DecRef(py_key);
         Py_DecRef(row);

         PyList_Append(ac_huff_tables, dict);
         Py_DecRef(dict);
      }
   }

   py_key = Py_BuildValue("s", "ac_huff_tables");
   PyDict_SetItem(result, py_key, ac_huff_tables);
   Py_DecRef(py_key);
   Py_DecRef(ac_huff_tables);
   // }}}

   // {{{ DC Huffman tables 
   PyObject* dc_huff_tables = PyList_New(0);
   assert(PyList_Check(dc_huff_tables));

   for(size_t n = 0; n < NUM_HUFF_TBLS; n++) 
   {
      if(cinfo.dc_huff_tbl_ptrs[n] != NULL) 
      {
         JHUFF_TBL *huff_ptr = cinfo.dc_huff_tbl_ptrs[n];

         dict = PyDict_New();
         assert(PyDict_Check(dict));

         row = PyList_New(0);
         assert(PyList_Check(row));
         for(size_t i=1; i<=16; i++)
            list_append_float(row, huff_ptr->bits[i]);
         
         py_key = Py_BuildValue("s", "counts");
         PyDict_SetItem(dict, py_key, row);
         Py_DecRef(py_key);
         Py_DecRef(row);

         row = PyList_New(0);
         assert(PyList_Check(row));
         for(size_t i=0; i<256; i++)
            list_append_float(row, huff_ptr->huffval[i]);
         
         py_key = Py_BuildValue("s", "symbols");
         PyDict_SetItem(dict, py_key, row);
         Py_DecRef(py_key);
         Py_DecRef(row);

         PyList_Append(dc_huff_tables, dict);
         Py_DecRef(dict);
      }
   }

   py_key = Py_BuildValue("s", "dc_huff_tables");
   PyDict_SetItem(result, py_key, dc_huff_tables);
   Py_DecRef(py_key);
   Py_DecRef(dc_huff_tables);
   // }}}

   // {{{ DCT coefficients
   jvirt_barray_ptr *coefs = jpeg_read_coefficients(&cinfo);

   PyObject* coef_arrays = PyList_New(0);
   assert(PyList_Check(coef_arrays));

   for(int ci=0; ci<cinfo.num_components; ci++) 
   {
      jpeg_component_info *compptr = cinfo.comp_info + ci;

      PyObject *py_blk_y = PyList_New(0);
      assert(PyList_Check(py_blk_y));
      
      for(size_t blk_y=0; blk_y<compptr->height_in_blocks; blk_y++)
      {
         PyObject *py_blk_x = PyList_New(0);
         assert(PyList_Check(py_blk_x));

         JBLOCKARRAY buffer = (cinfo.mem->access_virt_barray)
    	   ((j_common_ptr) &cinfo, coefs[ci], blk_y, 1, FALSE);
         for(size_t blk_x=0; blk_x<compptr->width_in_blocks; blk_x++)
         {
            row = PyList_New(0);
            assert(PyList_Check(row));
            JCOEFPTR bufptr = buffer[0][blk_x];
            for(size_t i=0; i<DCTSIZE; i++)
            {
               col = PyList_New(0);
               assert(PyList_Check(col));
               for (size_t j=0; j<DCTSIZE; j++)
               {
                  py_value = Py_BuildValue("f", (double) bufptr[i*DCTSIZE+j]);
                  PyList_Append(col, py_value);
                  Py_DecRef(py_value);
               }
               PyList_Append(row, col);
               Py_DecRef(col);
            }
            PyList_Append(py_blk_x, row);
            Py_DecRef(row);
         }
         PyList_Append(py_blk_y, py_blk_x);
         Py_DecRef(py_blk_x);
      }

      PyList_Append(coef_arrays, py_blk_y);
      Py_DecRef(py_blk_y);
   }

   py_key = Py_BuildValue("s", "coef_arrays");
   PyDict_SetItem(result, py_key, coef_arrays);
   Py_DecRef(py_key);
   Py_DecRef(coef_arrays);
   // }}}

   jpeg_finish_decompress(&cinfo);
   jpeg_destroy_decompress(&cinfo);
   fclose(f);

   PyGILState_Release(gstate);

   return result;
}
// }}}

// {{{ write_file()
LIBRARY_API void write_file(PyObject *data, const char *path)
{
   PyGILState_STATE gstate = PyGILState_Ensure();

   FILE *f = NULL;
   struct jpeg_compress_struct cinfo;
   struct my_error_mgr jerr;


   if((f = fopen(path, "wb")) == NULL)
   {
      fprintf(stderr, "Can not open file: %s\n", path);
      PyGILState_Release(gstate);
      return;
   }

   /* set up the normal JPEG error routines, then override error_exit. */
   cinfo.err = jpeg_std_error(&jerr.pub);
   jerr.pub.error_exit = my_error_exit;
   jerr.pub.output_message = my_output_message;

   /* establish the setjmp return context for my_error_exit to use. */
   if (setjmp(jerr.setjmp_buffer)) {
      jpeg_destroy_compress(&cinfo);
      fclose(f);
      fprintf(stderr, "Error writing to file: %s\n", path);
      PyGILState_Release(gstate);
      return;
   }

   /* initialize JPEG decompression object */
   jpeg_create_compress(&cinfo);

   /* Set the compression object with our parameters */
   cinfo.image_width = dict_get_int(data, "image_width");
   cinfo.image_height = dict_get_int(data, "image_height");
   cinfo.input_components = dict_get_int(data, "image_components");
   cinfo.in_color_space = dict_get_int(data, "image_color_space");


   /* write the output file */
   jpeg_stdio_dest(&cinfo, f);

   /* set default parameters */
   jpeg_set_defaults(&cinfo);

   /* set original density configuration..
      It must be set after jpeg_set_defaults() */
   cinfo.X_density = dict_get_int(data, "X_density");
   cinfo.Y_density = dict_get_int(data, "Y_density");
   cinfo.density_unit = dict_get_int(data, "density_unit");
   cinfo.block_size = dict_get_int(data, "block_size");
   cinfo.min_DCT_h_scaled_size = dict_get_int(data, "min_DCT_h_scaled_size");
   cinfo.min_DCT_v_scaled_size = dict_get_int(data, "min_DCT_v_scaled_size");

   //cinfo.optimize_coding = dict_get_int(data, "optimize_coding"); XXX
   cinfo.num_components = dict_get_int(data, "jpeg_components");
   cinfo.jpeg_color_space = dict_get_int(data, "jpeg_color_space");

   /* support for progressive mode */
   if(dict_get_int(data, "progressive_mode")) 
      jpeg_simple_progression(&cinfo);

   /* Component info */
   PyObject *comp_info = dict_get_object(data, "comp_info");
   for(int ci=0; ci<cinfo.num_components; ci++)
   {
      PyObject* item = PyList_GetItem(comp_info, ci);
      cinfo.comp_info[ci].component_id = dict_get_int(item, "component_id");
      cinfo.comp_info[ci].h_samp_factor = dict_get_int(item, "h_samp_factor");
      cinfo.comp_info[ci].v_samp_factor = dict_get_int(item, "v_samp_factor");
      cinfo.comp_info[ci].quant_tbl_no = dict_get_int(item, "quant_tbl_no");
      cinfo.comp_info[ci].ac_tbl_no = dict_get_int(item, "ac_tbl_no");
      cinfo.comp_info[ci].dc_tbl_no = dict_get_int(item, "dc_tbl_no");

      jpeg_component_info *compptr = cinfo.comp_info + ci;
      compptr->DCT_h_scaled_size = dict_get_int(item, "DCT_h_scaled_size");
      compptr->DCT_v_scaled_size = dict_get_int(item, "DCT_v_scaled_size");
   }


   /* DCT coefficients */
   PyObject *py_coef_arrays = dict_get_object(data, "coef_arrays");
   PyObject* py_tmp1 = PyList_GetItem(py_coef_arrays, 0);
   PyObject* py_tmp2 = PyList_GetItem(py_tmp1, 0);
   int height_in_blocks = PyList_Size(py_tmp1);
   int width_in_blocks = PyList_Size(py_tmp2);

   jvirt_barray_ptr *coef_arrays = (jvirt_barray_ptr *)
      (cinfo.mem->alloc_small) ((j_common_ptr) &cinfo, JPOOL_IMAGE,
            sizeof(jvirt_barray_ptr) * cinfo.num_components);
   for(int ci=0; ci<cinfo.num_components; ci++)
   {
      jpeg_component_info *compptr = cinfo.comp_info + ci;
      compptr->height_in_blocks = height_in_blocks;
      compptr->width_in_blocks = width_in_blocks;

      coef_arrays[ci] = (cinfo.mem->request_virt_barray)
         ((j_common_ptr) &cinfo, JPOOL_IMAGE, TRUE,
          (JDIMENSION) jround_up((long) compptr->width_in_blocks,
             (long) compptr->h_samp_factor),
          (JDIMENSION) jround_up((long) compptr->height_in_blocks,
             (long) compptr->v_samp_factor),
          (JDIMENSION) compptr->v_samp_factor);
   }

#if JPEG_LIB_VERSION >= 80
   cinfo.jpeg_width = cinfo.image_width;
   cinfo.jpeg_height = cinfo.image_height;
#endif

   jpeg_write_coefficients(&cinfo, coef_arrays);

   /* populate DCT coefficients */
   for(int ci=0; ci<cinfo.num_components; ci++)
   {
      PyObject* py_blk_x = PyList_GetItem(py_coef_arrays, ci);
      jpeg_component_info *compptr = cinfo.comp_info + ci;

      for(size_t blk_y = 0; blk_y < compptr->height_in_blocks; blk_y++)
      {
         PyObject* py_blk_y = PyList_GetItem(py_blk_x, blk_y);

         JBLOCKARRAY buffer = (cinfo.mem->access_virt_barray)
            ((j_common_ptr) &cinfo, coef_arrays[ci], blk_y, 1, TRUE);

         for(size_t blk_x = 0; blk_x < compptr->width_in_blocks; blk_x++)
         {
            PyObject* row = PyList_GetItem(py_blk_y, blk_x);

            JCOEFPTR bufptr = buffer[0][blk_x];
            for(size_t i = 0; i < DCTSIZE; i++)
            {
               PyObject* col = PyList_GetItem(row, i);
               for(size_t j = 0; j < DCTSIZE; j++)
               {
                  PyObject* item = PyList_GetItem(col, j);
                  double value = PyFloat_AsDouble(item);
                  bufptr[i*DCTSIZE+j] = (JCOEF) value;
               }
            }
         }
      }
   }


   /* Quantization tables */
   PyObject *quant_tables = dict_get_object(data, "quant_tables");
   ssize_t n;
   for(n=0; n<PyList_Size(quant_tables); n++)
   {
      if(cinfo.quant_tbl_ptrs[n] == NULL)
         cinfo.quant_tbl_ptrs[n] = jpeg_alloc_quant_table((j_common_ptr) &cinfo);

      PyObject* row = PyList_GetItem(quant_tables, n);
      for(size_t i=0; i<DCTSIZE; i++) 
      {
         PyObject* col = PyList_GetItem(row, i);
         for(size_t j=0; j<DCTSIZE; j++) 
         {
            PyObject* item = PyList_GetItem(col, j);
            int t = PyLong_AsLong(item);   
            if (t<1 || t>65535)
            {
               fprintf(stderr, "Quantization table entries not in range 1..65535");
               PyGILState_Release(gstate);
               return;
            }
            cinfo.quant_tbl_ptrs[n]->quantval[i*DCTSIZE+j] = (UINT16) t;
         }
      }
   }
   
   for(; n < NUM_QUANT_TBLS; n++)
      cinfo.quant_tbl_ptrs[n] = NULL;



   jpeg_finish_compress(&cinfo);
   jpeg_destroy_compress(&cinfo);
   fclose(f);

   PyGILState_Release(gstate);
}
// }}}

