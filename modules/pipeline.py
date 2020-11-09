import tensorflow as tf
import apache_beam as beam
import tensorflow_text as text 
import tensorflow_transform.beam as tft_beam
import tensorflow_transform.coders as tft_coders
from apache_beam.options.pipeline_options import PipelineOptions
import tempfile

model = None
force_tf_compat_v1 = False

# Does not work with or without force_tf_compat_v1 = False
# See the error logs in universal-sentence-encoder-large.logs
MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder-large/5"

def embed_text(text):
    import tensorflow_hub as hub
    global model
    if model is None:
        model = hub.load(MODEL_URL)
    return model(text)

def get_metadata():
    from tensorflow_transform.tf_metadata import dataset_schema
    from tensorflow_transform.tf_metadata import dataset_metadata

    metadata = dataset_metadata.DatasetMetadata(dataset_schema.Schema({
        "id": dataset_schema.ColumnSchema(
            tf.string, [], dataset_schema.FixedColumnRepresentation()),
        "text": dataset_schema.ColumnSchema(
            tf.string, [], dataset_schema.FixedColumnRepresentation())
    }))
    return metadata


def preprocess_fn(input_features):
    text_integerized = embed_text(input_features["text"])
    output_features = {
        "id": input_features["id"],
        "embedding": text_integerized
    }
    return output_features


def print_pass(input_features):
    # print and pass-through
    print(input_features)
    return input_features


def run(pipeline_options, known_args):
    global force_tf_compat_v1
    argv = None  # if None, uses sys.argv
    pipeline_options = PipelineOptions(argv)
    pipeline = beam.Pipeline(options=pipeline_options)

    with tft_beam.Context(temp_dir=tempfile.mkdtemp(), force_tf_compat_v1=force_tf_compat_v1):
        print("Context force_tf_compat_v1: {}".format(tft_beam.Context.get_use_tf_compat_v1()))
        articles = (
            pipeline
            | beam.Create([
                {"id": "01", "text": "To be, or not to be: that is the question: "},
                {"id": "02", "text": "Whether 'tis nobler in the mind to suffer "},
                {"id": "03", "text": "The slings and arrows of outrageous fortune, "},
                {"id": "04", "text": "Or to take arms against a sea of troubles, "},
            ])
        )

        articles_dataset = (articles, get_metadata())

        transformed_dataset, transform_fn = (
            articles_dataset
            | "Extract embeddings" >> tft_beam.AnalyzeAndTransformDataset(preprocess_fn)
        )

        transformed_data, transformed_metadata = transformed_dataset

        _ = (
            transformed_data
            | "Print embeddings" >> beam.Map(print_pass)
            | "Write embeddings to TFRecords" >> beam.io.tfrecordio.WriteToTFRecord(
                file_path_prefix="{0}".format(known_args.output_dir),
                file_name_suffix=".tfrecords",
                coder=tft_coders.example_proto_coder.ExampleProtoCoder(
                    transformed_metadata.schema),
                num_shards=1)
        )

    job = pipeline.run()
    if pipeline_options.get_all_options()["runner"] == "DirectRunner":
        job.wait_until_finish()
