// This is a script to upgrade "V0" network prototxts to the new format.
// Usage:
//    upgrade_net_proto_binary v0_net_proto_file_in net_proto_file_out

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

  NetParameter net_param;
  string input_filename(argv[1]);
  if (!ReadProtoFromBinaryFile(input_filename, &net_param)) {
    return 2;
  }
  bool need_upgrade = NetNeedsUpgrade(net_param);
  bool success = true;
  if (need_upgrade) {
    success = UpgradeNetAsNeeded(input_filename, &net_param);
    if (!success) {
        ;;
    }
  } else {
      ;;
  }

  WriteProtoToBinaryFile(net_param, argv[2]);

  return !success;
}
