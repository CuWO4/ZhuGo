#include <Python.h>

#define NDEBUG

#include "board.c"

// Python wrapper for new_board
static PyObject* py_new_board(PyObject* self, PyObject* args) {
  int rows, cols;
  if (!PyArg_ParseTuple(args, "ii", &rows, &cols)) {
    return NULL;
  }
  Board* board = new_board(rows, cols);
  return PyCapsule_New(board, "Board", NULL);
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
  return PyCapsule_New(cloned, "Board", NULL);
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
