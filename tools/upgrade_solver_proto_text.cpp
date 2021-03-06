// This is a script to upgrade old solver prototxts to the new format.
// Usage:
//    upgrade_solver_proto_text old_solver_proto_file_in solver_proto_file_out

#include <cstring>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

using std::ofstream;

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc != 3) {
    return 1;
  }

  SolverParameter solver_param;
  string input_filename(argv[1]);
  if (!ReadProtoFromTextFile(input_filename, &solver_param)) {
    return 2;
  }
  bool need_upgrade = SolverNeedsTypeUpgrade(solver_param);
  bool success = true;
  if (need_upgrade) {
    success = UpgradeSolverAsNeeded(input_filename, &solver_param);
    if (!success) {
        ;;
    }
  } else {
      ;;
  }

  // Save new format prototxt.
  WriteProtoToTextFile(solver_param, argv[2]);
  return !success;
}
