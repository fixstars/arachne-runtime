#include <pybind11/pybind11.h>

#include <iostream>
#include <memory>

static int ID = 0;

class CppRuntime {
 public:
  CppRuntime() {
    id_ = ID++;
    pybind11::print("CppRuntime::Init", id_);
    internal_data_ = std::make_unique<int[]>(3);
  }

  ~CppRuntime() {
    //  DO NOT FREE ANY RESOURCE HERE
    //  Plese implement an indepented function like CppRuntime::Done() and call
    //  the function to free any resource allocated via this object. This
    //  destructor will be called when the Python Garbage Collector cleans up an
    //  object corresponding to this object. When the reference counter for the
    //  object gets to zero, the garbage collector may destroy the object (and
    //  thereby invoke this destructor) but it doesn't have to do it at that
    //  moment.
    pybind11::print("CppRuntime::Delete", id_);
  }

  void Done() {
    pybind11::print("CppRuntime::Done", id_);
    // Write the release procedures here
    internal_data_.release();
  }

  void SetInput() { pybind11::print("CppRuntime::SetInput", id_); }

  void Run() { pybind11::print("CppRuntime::Run", id_); }

  void GetOutput() { pybind11::print("CppRuntime::GetOutput", id_); }

 private:
  int id_;
  std::unique_ptr<int[]> internal_data_;
};

PYBIND11_MODULE(cppruntime, m) {
  pybind11::class_<CppRuntime>(m, "CppRuntime")
      .def(pybind11::init<>())
      .def("done", &CppRuntime::Done)
      .def("run", &CppRuntime::Run)
      .def("set_input", &CppRuntime::SetInput)
      .def("get_output", &CppRuntime::GetOutput);
}
