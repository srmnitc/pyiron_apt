from pyiron_workflow import Workflow

@Workflow.wrap.as_function_node('visualize_hdf')
def visualize_hdf(results_file_path):
    from jupyterlab_h5web import H5Web
    return H5Web(results_file_path)