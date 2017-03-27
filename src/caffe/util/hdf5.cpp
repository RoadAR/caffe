#include "caffe/util/hdf5.hpp"

#include <string>
#include <vector>

namespace caffe {

// Verifies format of data stored in HDF5 file and reshapes blob accordingly.
template <typename Dtype>
void hdf5_load_nd_dataset_helper(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob) {
  // Verify that the dataset exists.
  H5LTfind_dataset(file_id, dataset_name_);
  // Verify that the number of dimensions is in the accepted range.
  int ndims;
  H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);

  // Verify that the data format is what we expect: float or double.
  std::vector<hsize_t> dims(ndims);
  H5T_class_t class_;
  H5LTget_dataset_info(file_id, dataset_name_, dims.data(), &class_, NULL);
  switch (class_) {
  case H5T_FLOAT:
    //{ LOG_FIRST_N(INFO, 1) << "Datatype class: H5T_FLOAT"; }
    break;
  case H5T_INTEGER:
    //{ LOG_FIRST_N(INFO, 1) << "Datatype class: H5T_INTEGER"; }
    break;
  case H5T_TIME:
      ;;
  case H5T_STRING:
      ;;
  case H5T_BITFIELD:
      ;;
  case H5T_OPAQUE:
      ;;
  case H5T_COMPOUND:
      ;;
  case H5T_REFERENCE:
      ;;
  case H5T_ENUM:
      ;;
  case H5T_VLEN:
      ;;
  case H5T_ARRAY:
      ;;
  default:
      ;;
  }

  vector<int> blob_dims(dims.size());
  for (int i = 0; i < dims.size(); ++i) {
    blob_dims[i] = dims[i];
  }
  blob->Reshape(blob_dims);
}

template <>
void hdf5_load_nd_dataset<float>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<float>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  H5LTread_dataset_float(file_id, dataset_name_, blob->mutable_cpu_data());
}

template <>
void hdf5_load_nd_dataset<double>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<double>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  H5LTread_dataset_double(file_id, dataset_name_, blob->mutable_cpu_data());
}

template <>
void hdf5_save_nd_dataset<float>(
    const hid_t file_id, const string& dataset_name, const Blob<float>& blob,
    bool write_diff) {
  int num_axes = blob.num_axes();
  hsize_t *dims = new hsize_t[num_axes];
  for (int i = 0; i < num_axes; ++i) {
    dims[i] = blob.shape(i);
  }
  const float* data;
  if (write_diff) {
    data = blob.cpu_diff();
  } else {
    data = blob.cpu_data();
  }
  H5LTmake_dataset_float(file_id, dataset_name.c_str(), num_axes, dims, data);
  delete[] dims;
}

template <>
void hdf5_save_nd_dataset<double>(
    hid_t file_id, const string& dataset_name, const Blob<double>& blob,
    bool write_diff) {
  int num_axes = blob.num_axes();
  hsize_t *dims = new hsize_t[num_axes];
  for (int i = 0; i < num_axes; ++i) {
    dims[i] = blob.shape(i);
  }
  const double* data;
  if (write_diff) {
    data = blob.cpu_diff();
  } else {
    data = blob.cpu_data();
  }
  H5LTmake_dataset_double(file_id, dataset_name.c_str(), num_axes, dims, data);
  delete[] dims;
}

string hdf5_load_string(hid_t loc_id, const string& dataset_name) {
  // Get size of dataset
  size_t size;
  H5T_class_t class_;
  H5LTget_dataset_info(loc_id, dataset_name.c_str(), NULL, &class_, &size);
  char *buf = new char[size];
  H5LTread_dataset_string(loc_id, dataset_name.c_str(), buf);
  string val(buf);
  delete[] buf;
  return val;
}

void hdf5_save_string(hid_t loc_id, const string& dataset_name,
                      const string& s) {
  H5LTmake_dataset_string(loc_id, dataset_name.c_str(), s.c_str());
}

int hdf5_load_int(hid_t loc_id, const string& dataset_name) {
  int val;
  H5LTread_dataset_int(loc_id, dataset_name.c_str(), &val);
  return val;
}

void hdf5_save_int(hid_t loc_id, const string& dataset_name, int i) {
  hsize_t one = 1;
  H5LTmake_dataset_int(loc_id, dataset_name.c_str(), 1, &one, &i);
}

int hdf5_get_num_links(hid_t loc_id) {
  H5G_info_t info;
  H5Gget_info(loc_id, &info);
  return info.nlinks;
}

string hdf5_get_name_by_idx(hid_t loc_id, int idx) {
  ssize_t str_size = H5Lget_name_by_idx(
      loc_id, ".", H5_INDEX_NAME, H5_ITER_NATIVE, idx, NULL, 0, H5P_DEFAULT);
  char *c_str = new char[str_size+1];
  string result(c_str);
  delete[] c_str;
  return result;
}

}  // namespace caffe
