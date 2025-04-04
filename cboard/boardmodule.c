#include <Python.h>

#include "board.h"
#include "ko.h"

void py_destructor(PyObject* capsule) {
  Board* board = PyCapsule_GetPointer(capsule, "Board");
  delete_board(board);
}

// Python wrapper for new_board
static PyObject* py_new_board(PyObject* self, PyObject* args) {
  int rows, cols;
  if (!PyArg_ParseTuple(args, "ii", &rows, &cols)) {
    return NULL;
  }
  Board* board = new_board(rows, cols);
  return PyCapsule_New(board, "Board", py_destructor);
}

// Python wrapper for delete_board
static PyObject* py_delete_board(PyObject* self, PyObject* args) {
  PyObject* board_capsule;
  if (!PyArg_ParseTuple(args, "O", &board_capsule)) {
    return NULL;
  }
  Board* board = PyCapsule_GetPointer(board_capsule, "Board");
  delete_board(board);
  Py_RETURN_NONE;
}

// Python wrapper for clone_board
static PyObject* py_clone_board(PyObject* self, PyObject* args) {
  PyObject* board_capsule;
  if (!PyArg_ParseTuple(args, "O", &board_capsule)) {
    return NULL;
  }
  Board* board = PyCapsule_GetPointer(board_capsule, "Board");
  Board* cloned = clone_board(board);
  return PyCapsule_New(cloned, "Board", py_destructor);
}

// Python wrapper for get_piece
static PyObject* py_get_piece(PyObject* self, PyObject* args) {
  PyObject* board_capsule;
  int row, col;
  if (!PyArg_ParseTuple(args, "Oii", &board_capsule, &row, &col)) {
    return NULL;
  }
  Board* board = PyCapsule_GetPointer(board_capsule, "Board");
  Piece piece = get_piece(board, row, col);
  return PyLong_FromLong(piece);
}

// Python wrapper for get_qi
static PyObject* py_get_qi(PyObject* self, PyObject* args) {
  PyObject* board_capsule;
  int row, col;
  if (!PyArg_ParseTuple(args, "Oii", &board_capsule, &row, &col)) {
    return NULL;
  }
  Board* board = PyCapsule_GetPointer(board_capsule, "Board");
  unsigned qi = get_qi(board, row, col);
  return PyLong_FromUnsignedLong(qi);
}

// Python wrapper for get_random_qi_pos
static PyObject* py_get_random_qi_pos(PyObject* self, PyObject* args) {
  PyObject* board_capsule;
  int row, col;
  if (!PyArg_ParseTuple(args, "Oii", &board_capsule, &row, &col)) {
    return NULL;
  }
  Board* board = PyCapsule_GetPointer(board_capsule, "Board");
  unsigned pos_row, pos_col;
  get_random_qi_pos(&pos_row, &pos_col, board, row, col);

  PyObject* pos_tuple = PyTuple_New(2);
  PyTuple_SetItem(pos_tuple, 0, PyLong_FromLong(pos_row));
  PyTuple_SetItem(pos_tuple, 1, PyLong_FromLong(pos_col));
  return pos_tuple;
}

// Python wrapper for is_valid_move
static PyObject* py_is_valid_move(PyObject* self, PyObject* args) {
  PyObject* board_capsule;
  int row, col;
  int player;  // Assume player is passed as an integer (0 = EMPTY, 1 = BLACK, 2 = WHITE)
  if (!PyArg_ParseTuple(args, "Oiii", &board_capsule, &row, &col, &player)) {
    return NULL;
  }
  Board* board = PyCapsule_GetPointer(board_capsule, "Board");
  bool valid = is_valid_move(board, row, col, player);
  return PyBool_FromLong(valid);
}

// Python wrapper for place_piece
static PyObject* py_place_piece(PyObject* self, PyObject* args) {
  PyObject* board_capsule;
  int row, col;
  int player;  // Assume player is passed as an integer (0 = EMPTY, 1 = BLACK, 2 = WHITE)
  if (!PyArg_ParseTuple(args, "Oiii", &board_capsule, &row, &col, &player)) {
    return NULL;
  }
  Board* board = PyCapsule_GetPointer(board_capsule, "Board");
  place_piece(board, row, col, player);
  Py_RETURN_NONE;
}

// Python wrapper for hash (total board hash)
static PyObject* py_hash(PyObject* self, PyObject* args) {
  PyObject* board_capsule;
  if (!PyArg_ParseTuple(args, "O", &board_capsule)) {
    return NULL;
  }
  Board* board = PyCapsule_GetPointer(board_capsule, "Board");
  uint64_t hash_value = hash(board);
  return PyLong_FromUnsignedLongLong(hash_value);
}

#ifndef MAX_BUFFER_SIZE
#define MAX_BUFFER_SIZE 2048
#endif
static PyObject* py_serialize(PyObject* self, PyObject* args) {
  PyObject* board_capsule;
  if (!PyArg_ParseTuple(args, "O", &board_capsule)) {
    return NULL;
  }

  Board* board = PyCapsule_GetPointer(board_capsule, "Board");

  static char buf[MAX_BUFFER_SIZE];
  size_t size = serialize(board, buf);

  return PyBytes_FromStringAndSize(buf, size);
}

static PyObject* py_deserialize(PyObject* self, PyObject* args) {
  PyObject* py_buf;
  if (!PyArg_ParseTuple(args, "O", &py_buf)) {
    return NULL;
  }

  char* buf = PyBytes_AsString(py_buf);

  Board* board = deserialize(buf);
  return PyCapsule_New(board, "Board", py_destructor);
}
#undef MAX_BUFFER_SIZE

static PyObject* py_does_violate_ko(PyObject* self, PyObject* args) {
  PyObject* board_capsule;
  PyObject* last_board_capsule;
  int player;
  int row, col;

  if (!PyArg_ParseTuple(args, "OiiiO", &board_capsule, &player, &row, &col, &last_board_capsule)) {
    return NULL;
  }

  Board* board = PyCapsule_GetPointer(board_capsule, "Board");
  Board* last_board = PyCapsule_GetPointer(last_board_capsule, "Board");

  if (!board || !last_board) {
    return NULL;
  }

  if (does_violate_ko(board, player, row, col, last_board)) {
    Py_RETURN_TRUE;
  }
  else {
    Py_RETURN_FALSE;
  }
}

// Methods table
static PyMethodDef BoardMethods[] = {
  {"new_board", py_new_board, METH_VARARGS, "Create a new board"},
  {"delete_board", py_delete_board, METH_VARARGS, "Delete a board"},
  {"clone_board", py_clone_board, METH_VARARGS, "Clone a board"},
  {"get_piece", py_get_piece, METH_VARARGS, "Get the piece at a position"},
  {"get_qi", py_get_qi, METH_VARARGS, "Get liberties for a position"},
  {"get_random_qi_pos", py_get_random_qi_pos, METH_VARARGS, "get a random qi position tuple"},
  {"is_valid_move", py_is_valid_move, METH_VARARGS, "Check if a move is valid"},
  {"place_piece", py_place_piece, METH_VARARGS, "Place a piece on the board"},
  {"hash", py_hash, METH_VARARGS, "get zobrist hash of board"},
  {"serialize", py_serialize, METH_VARARGS, "Serialize the board"},
  {"deserialize", py_deserialize, METH_VARARGS, "Deserialize the board"},
  {"does_violate_ko", py_does_violate_ko, METH_VARARGS, "check ko violation"},
  {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef boardmodule = {
  PyModuleDef_HEAD_INIT,
  "cboard",
  "C Board module for Go",
  -1,
  BoardMethods
};

PyMODINIT_FUNC PyInit_boardmodule(void) {
  return PyModule_Create(&boardmodule);
}
