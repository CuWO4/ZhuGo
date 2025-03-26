#include <Python.h>

#include "encoder.h"
#include <numpy/arrayobject.h>

static PyObject* py_encode(PyObject* self, PyObject* args) {
  PyObject* py_board;
  PyObject* py_last_board;
  int player;

  if (!PyArg_ParseTuple(args, "OOi", &py_board, &py_last_board, &player)) {
    return NULL;
  }

  Board* board = static_cast<Board*>(PyCapsule_GetPointer(py_board, "Board"));
  if (board == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid Board capsule for 'board'");
    return NULL;
  }

  Board* last_board = static_cast<Board*>(PyCapsule_GetPointer(py_last_board, "Board"));
  if (last_board == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid Board capsule for 'last_board'");
    return NULL;
  }

  constexpr int num_dims = 3;
  npy_intp dims[] = { channels, board->rows, board->cols }; 

  float* data = encode(board, last_board, player);

  PyObject* numpy_array = PyArray_SimpleNewFromData(num_dims, dims, NPY_FLOAT32, (void*)data);

  if (numpy_array == NULL) {
    free(data);
    return NULL;
  }

  PyArrayObject* arr = (PyArrayObject*)numpy_array;
  PyArray_FLAGS(arr) |= NPY_ARRAY_OWNDATA;

  return numpy_array;
}

static PyMethodDef EncoderMethods[] = {
  {"encode", py_encode, METH_VARARGS, "Encode a board to numpy array"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef boardmodule = {
  PyModuleDef_HEAD_INIT,
  "cencoder",
  "C Encoder for ZhuGo",
  -1,
  EncoderMethods
};

PyMODINIT_FUNC PyInit_encodermodule(void) {
  import_array();
  return PyModule_Create(&boardmodule);
}
