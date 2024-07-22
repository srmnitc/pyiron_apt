from pyiron_workflow import Workflow

@Workflow.wrap.as_function_node('cspace_voxelize')
def cspace_voxelize(config_file_path, 
                    results_file_path,
                    recon_file_path,
                    range_file_path,
                    entry_id=1, 
                    verbose=False):
    from compositionspace.preparation import ProcessPreparation
    voxelize = ProcessPreparation(config_file_path, 
                                  results_file_path, 
                                  entry_id=1, 
                                  verbose=False)
    voxelize.run(recon_file_path=recon_file_path,
                range_file_path=range_file_path)
    return results_file_path

@Workflow.wrap.as_function_node('cspace_autophase')
def cspace_autophase(config_file_path, 
                    results_file_path, 
                    entry_id=1, 
                    verbose=False):
    from compositionspace.autophase import ProcessAutomatedPhaseAssignment
    autophase = ProcessAutomatedPhaseAssignment(config_file_path, 
                                            results_file_path, 
                                            entry_id=1, 
                                            verbose=False)
    autophase.run()
    return results_file_path

@Workflow.wrap.as_function_node('cspace_segmentation')
def cspace_segmentation(config_file_path, 
                    results_file_path, 
                    entry_id=1, 
                    verbose=False):
    from compositionspace.segmentation import ProcessSegmentation
    segmentation = ProcessSegmentation(config_file_path, 
                                   results_file_path, 
                                   entry_id=1, 
                                   verbose=False)
    segmentation.run()
    return results_file_path

@Workflow.wrap.as_function_node('cspace_clustering')
def cspace_clustering(config_file_path, 
                    results_file_path, 
                    entry_id=1, 
                    verbose=False):
    from compositionspace.clustering import ProcessClustering
    clustering = ProcessClustering(config_file_path, 
                               results_file_path, 
                               entry_id=1, 
                               verbose=False)
    clustering.run()
    return results_file_path

@Workflow.wrap.as_function_node('generate_xdmf')
def generate_xdmf(results_file_path):
    from compositionspace.visualization import generate_xdmf_for_visualizing_content
    generate_xdmf_for_visualizing_content(results_file_path)
    return results_file_path + '.xdmf'