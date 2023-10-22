
import gradio as gr

from ebsynth_utility import ebsynth_utility_process
from modules import script_callbacks
from modules.call_queue import wrap_gradio_gpu_call

def on_ui_tabs():

    with gr.Blocks(analytics_enabled=False) as ebs_interface_lite:
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'):

                with gr.Row():
                    with gr.Tabs(elem_id="ebs_settings"):
                        with gr.TabItem('Project Setting', elem_id='ebs_project_setting'):
                            project_dir = gr.Textbox(label='Project directory', lines=1)
                            original_movie_path = gr.Textbox(label='Original Movie Path', lines=1)

                            org_video = gr.Video(interactive=True, mirror_webcam=False)
                            def fn_upload_org_video(video):
                                return video
                            org_video.upload(fn_upload_org_video, org_video, original_movie_path)
                            gr.HTML(value="<p style='margin-bottom: 1.2em'>\
                                    If you have trouble entering the video path manually, you can also use drag and drop \
                                    </p>")

                        with gr.TabItem('Configuration', elem_id='ebs_configuration'):
                            with gr.Tabs(elem_id="ebs_configuration_tab"):
                                with gr.TabItem(label="Stage 1",elem_id='ebs_configuration_tab1'):
                                    with gr.Row():
                                        frame_width = gr.Number(value=-1, label="Frame Width", precision=0, interactive=True)
                                        frame_height = gr.Number(value=-1, label="Frame Height", precision=0, interactive=True)
                                    gr.HTML(value="<p style='margin-bottom: 1.2em'>\
                                            -1 means that it is calculated automatically. If both are -1, the size will be the same as the source size. \
                                            </p>")
                                    # with gr.Accordion(label="mask options",open=False):

                                    # mask_mode =gr.Radio(label='mask mode', choices=["Normal","Invert","None"], value="None")
                                    st1_masking_method_index = gr.Radio(label='Masking Method', choices=["transparent-background","clipseg","transparent-background AND clipseg"], value="transparent-background", type="index")

                                    with gr.Accordion(label="transparent-background options"):
                                        st1_mask_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Mask Threshold', value=0.0)

                                        # https://pypi.org/project/transparent-background/
                                        gr.HTML(value="<p style='margin-bottom: 0.7em'>\
                                                configuration for \
                                                <font color=\"blue\"><a href=\"https://pypi.org/project/transparent-background\">[transparent-background]</a></font>\
                                                </p>")
                                        tb_use_fast_mode = gr.Checkbox(label="Use Fast Mode(It will be faster, but the quality of the mask will be lower.)", value=False)
                                        tb_use_jit = gr.Checkbox(label="Use Jit", value=False)

                                    with gr.Accordion(label="clipseg options"):
                                        clipseg_mask_prompt = gr.Textbox(label='Mask Target (e.g., girl, cats)', lines=1)
                                        clipseg_exclude_prompt = gr.Textbox(label='Exclude Target (e.g., finger, book)', lines=1)
                                        clipseg_mask_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Mask Threshold', value=0.4)
                                        clipseg_mask_blur_size = gr.Slider(minimum=0, maximum=150, step=1, label='Mask Blur Kernel Size(MedianBlur)', value=11)
                                        clipseg_mask_blur_size2 = gr.Slider(minimum=0, maximum=150, step=1, label='Mask Blur Kernel Size(GaussianBlur)', value=11)

                                with gr.TabItem(label="Stage 2", elem_id='ebs_configuration_tab2'):
                                    key_min_gap = gr.Slider(minimum=0, maximum=500, step=1, label='Minimum keyframe gap', value=10)
                                    key_max_gap = gr.Slider(minimum=0, maximum=1000, step=1, label='Maximum keyframe gap', value=300)
                                    key_th = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, label='Threshold of delta frame edge', value=8.5)
                                    custom_frames = gr.Slider(minimum=0, maximum=100, step=1, label='Custom frames [In development]', value=0)
                                    key_add_last_frame = gr.Checkbox(label="Add last frame to keyframes", value=True)
                                
                                with gr.TabItem(label="Stage 3 [In development]", elem_id='ebs_configuration_tab3'):
                                    grid_width = gr.Slider(minimum=2, maximum=3, step=1, label='Width', value=2)
                                    grid_height = gr.Slider(minimum=2, maximum=3, step=1, label='Height', value=2)

                                with gr.TabItem(label="Stage 5", elem_id='ebs_configuration_tab5'):
                                     key_weight = gr.Textbox(label='Key Weight', lines=1,value=1.0)
                                     video_weight = gr.Textbox(label='Video Weight', lines=1,value=4.0)
                                     mask_weight = gr.Textbox(label='Mask Weight', lines=1,value=1.0)
                                     adv_mapping = gr.Textbox(label='Mapping', lines=1,value=10.0)
                                     adv_de_flicker = gr.Textbox(label='De-flicker', lines=1,value=1.0)
                                     dv_diversity = gr.Textbox(label='Diversity', lines=1,value=3500.0)
                                     adv_detail = gr.Radio(label='Synthesis Detail', choices=["None","High","Medium","Low","Carbage"], value="High", type="index")
                                     adv_gpu = gr.Checkbox(label='Use GPU',value=True,lines=3)

                                with gr.TabItem(label="Stage 6", elem_id='ebs_configuration_tab6'):
                                    blend_rate = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Crossfade blend rate', value=1.0)
                                    export_type = gr.Dropdown(choices=["mp4","webm","gif","rawvideo"], value="mp4", label="Export type")

                                with gr.TabItem(label="Stage 7", elem_id='ebs_configuration_tab7'):
                                    bg_src = gr.Textbox(label='Background source(mp4 or directory containing images)', lines=1)
                                    bg_type = gr.Dropdown(choices=["Fit video length","Loop"], value="Fit video length", label="Background type")
                                    mask_blur_size = gr.Slider(minimum=0, maximum=150, step=1, label='Mask Blur Kernel Size', value=5)
                                    mask_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Mask Threshold', value=0.0)
                                    #is_transparent = gr.Checkbox(label="Is Transparent", value=True, visible = False)
                                    fg_transparency = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Foreground Transparency', value=0.0)

                                with gr.TabItem(label="etc", elem_id='ebs_configuration_tab_etc'):
                                    mask_mode = gr.Dropdown(choices=["Normal","Invert","None"], value="Normal" ,label="Mask Mode")
                        with gr.TabItem('Info', elem_id='ebs_info'):
                            gr.HTML(value="<p style='margin-bottom: 0.7em'>\
                                                The process of creating a video can be divided into the following stages<br>\
                                                (Stage 3 and 4 only show a guide and do nothing actual processing)<br><br>\
                                                <b>Stage 1</b> <br>\
                                                    Extract frames from the original video. <br>\
                                                    Generate a mask image. <br><br>\
                                                <b>Stage 2</b> <br>\
                                                    Select keyframes to be given to ebsynth.<br><br>\
                                                <b>Stage 3 [In development]</b> <br>\
                                                    Create GRID 2x2 or 3x3<br><br>\
                                                <b>Stage 4 [In development]</b> <br>\
                                                    and upscale to the size of the original video.<br><br>\
                                                <b>Stage 5</b> <br>\
                                                    Rename keyframes.<br>\
                                                    Generate .ebs file (ebsynth project file)<br><br>\
                                                    \
                                                    Running Ebsynth (on your self)<br>\
                                                    Open the generated .ebs under project directory and press [Run All] button. <br>\
                                                    If ""out-*"" directory already exists in the Project directory, delete it manually before executing.<br>\
                                                    If multiple .ebs files are generated, run them all.<br><br>\
                                                <b>Stage 6</b> <br>\
                                                    Concatenate each frame while crossfading.<br>\
                                                    Composite audio files extracted from the original video onto the concatenated video.<br><br>\
                                                <b>Stage 7</b> <br>\
                                                    This is an extra stage.<br>\
                                                    You can put any image or images or video you like in the background.<br>\
                                                    You can specify in this field -> [Ebsynth Utility]->[configuration]->[stage 8]->[Background source]<br>\
                                                    If you have already created a background video in Invert Mask Mode([Ebsynth Utility]->[configuration]->[etc]->[Mask Mode]),<br>\
                                                    You can specify \"path_to_project_dir/inv/crossfade_tmp\"<br>\
                                                </p>")

                    with gr.Column(variant='panel'):
                        with gr.Column(scale=1):
                            with gr.Row():
                                stage_index = gr.Radio(label='Process Stage', choices=["Stage 1","Stage 2","In development","In development","Stage 5","Stage 6","Stage 7"], value="Stage 1", type="index", elem_id='ebs_stages')
                            
                            with gr.Row():
                                generate_btn = gr.Button('Generate', elem_id="ebs_generate_btn", variant='primary')
                            
                            with gr.Group():
                                debug_info = gr.HTML(elem_id="ebs_info_area", value=".")

                            with gr.Column(scale=2):
                                html_info = gr.HTML()                                                                                

            ebs_args = dict(
                fn=wrap_gradio_gpu_call(ebsynth_utility_process),
                inputs=[
                    stage_index,

                    project_dir,
                    original_movie_path,

                    frame_width,
                    frame_height,
                    st1_masking_method_index,
                    st1_mask_threshold,
                    tb_use_fast_mode,
                    tb_use_jit,
                    clipseg_mask_prompt,
                    clipseg_exclude_prompt,
                    clipseg_mask_threshold,
                    clipseg_mask_blur_size,
                    clipseg_mask_blur_size2,

                    key_min_gap,
                    key_max_gap,
                    key_th,
                    key_add_last_frame,

                    blend_rate,
                    export_type,

                    bg_src,
                    bg_type,
                    mask_blur_size,
                    mask_threshold,
                    fg_transparency,

                    mask_mode,

                    key_weight,
                    video_weight ,
                    mask_weight ,
                    adv_mapping,
                    adv_de_flicker,
                    dv_diversity,
                    adv_detail,
                    adv_gpu,

                ],
                outputs=[
                    debug_info,
                    html_info,
                ],
                show_progress=False,
            )
            generate_btn.click(**ebs_args)
           
    return (ebs_interface_lite, "Ebsynth Utility Lite", "ebs_interface_lite"),

script_callbacks.on_ui_tabs(on_ui_tabs)
