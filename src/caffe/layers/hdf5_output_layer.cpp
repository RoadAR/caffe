#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layers/hdf5_output_layer.hpp"
#include "caffe/util/hdf5.hpp"

namespace caffe {

template <typename Dtype>
void HDF5OutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  file_name_ = this->layer_param_.hdf5_output_param().file_name();
  file_id_ = H5Fcreate(file_name_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                       H5P_DEFAULT);
  file_opened_ = true;
}

template <typename Dtype>
HDF5OutputLayer<Dtype>::~HDF5OutputLayer<Dtype>() {
  if (file_opened_) {
    herr_t status = H5Fclose(file_id_);
  }
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::SaveBlobs() {
  // TODO: no limit on the number of blobs
  hdf5_save_nd_dataset(file_id_, HDF5_DATA_DATASET_NAME, data_blob_);
  hdf5_save_nd_dataset(file_id_, HDF5_DATA_LABEL_NAME, label_blob_);
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  data_blob_.Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  label_blob_.Reshape(bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
  const int data_datum_dim = bottom[0]->count() / bottom[0]->num();
  const int label_datum_dim = bottom[1]->count() / bottom[1]->num();

  for (int i = 0; i < bottom[0]->num(); ++i) {
    caffe_copy(data_datum_dim, &bottom[0]->cpu_data()[i * data_datum_dim],
        &data_blob_.mutable_cpu_data()[i * data_datum_dim]);
    caffe_copy(label_datum_dim, &bottom[1]->cpu_data()[i * label_datum_dim],
        &label_blob_.mutable_cpu_data()[i * label_datum_dim]);
  }
  SaveBlobs();
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

#ifdef CPU_ONLY
STUB_GPU(HDF5OutputLayer);
#endif

INSTANTIATE_CLASS(HDF5OutputLayer);
REGISTER_LAYER_CLASS(HDF5Output);

}  // namespace caffe
