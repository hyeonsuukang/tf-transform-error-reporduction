# Configurable parameters
ROOT_DIR="./data"

# Datastore parameters
KIND="test"

# Directory for output data files
OUTPUT_PREFIX="${ROOT_DIR}/${KIND}/embeddings/embed"

# Working directories for Dataflow
DF_JOB_DIR="${ROOT_DIR}/${KIND}/dataflow"
STAGING_LOCATION="${DF_JOB_DIR}/staging"
TEMP_LOCATION="${DF_JOB_DIR}/temp"

# Working directories for tf.transform
TRANSFORM_ROOT_DIR="${DF_JOB_DIR}/transform"
TRANSFORM_TEMP_DIR="${TRANSFORM_ROOT_DIR}/temp"
TRANSFORM_EXPORT_DIR="${TRANSFORM_ROOT_DIR}/export"

# Working directories for Debug log
DEBUG_OUTPUT_PREFIX="${DF_JOB_DIR}/debug/log"

# Running Config for Dataflow
RUNNER="DirectRunner"

echo "Running the Dataflow job..."

# Command to run the Dataflow job
python run_pipeline.py \
  --output_dir="${OUTPUT_PREFIX}" \
  --transform_temp_dir="${TRANSFORM_TEMP_DIR}" \
  --transform_export_dir="${TRANSFORM_EXPORT_DIR}" \
  --runner="${RUNNER}" \
  --kind="${KIND}" \
  --staging_location="${STAGING_LOCATION}" \
  --temp_location="${TEMP_LOCATION}" \
  --setup_file=$(pwd)/setup.py \
  --enable_debug \
  --debug_output_prefix="${DEBUG_OUTPUT_PREFIX}"