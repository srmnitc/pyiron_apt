from pyiron_workflow import Workflow

@Workflow.wrap.as_function_node('paraprobe_transcoder)
def transcoder(apt_file, rng_file, jobid=1):
    from paraprobe_parmsetup.transcoder_config import ParmsetupTranscoder, TranscodingTask
    from paraprobe_transcoder import ParaprobeTranscoder
    
    transcoder = ParmsetupTranscoder()
    transcoder_config = transcoder.load_reconstruction_and_ranging(
        recon_fpath=apt_file,
        range_fpath=rng_file,
        jobid=jobid)
    transcoder = ParaprobeTranscoder(transcoder_config)
    results = transcoder.execute()
    return results
    