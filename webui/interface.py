# -*- coding: utf-8 -*-
"""
Main interface rendering for SoulX-Podcast WebUI.
"""

import os
import gradio as gr

from .constants import MAX_SPEAKERS, MAX_TEXT_INPUTS
from .i18n import i18n, get_i18n_dict, get_speaker_display_label, get_language
from .file_manager import list_config_files
from .components import create_speaker_group
from .callbacks import (
    add_speaker,
    quick_add_speakers,
    batch_delete_speakers,
    select_all_checkboxes,
    select_none_checkboxes,
    update_text_inputs_visibility,
    collect_and_synthesize_queue,
    change_component_language,
    update_single_speaker_label,
    update_speaker_accordion_label,
    update_speaker_selection_choices,
    selection_group_to_checkboxes,
    select_all_selection_group,
    select_none_selection_group,
    refresh_all_speaker_labels_after_load,
)
from .config_manager import (
    export_current_config,
    refresh_config_dropdown,
    load_uploaded_and_apply,
    load_selected_and_apply,
)

# Custom CSS for better UI
CSS = """
.container { max_width: 1400px; margin: auto; }
.header-row { align-items: center; margin-bottom: 20px; border-bottom: 1px solid #eee; padding-bottom: 10px; }
.header-logo { height: 50px; object-fit: contain; }
.section-header { margin-top: 10px; margin-bottom: 5px; font-size: 1.1em; font-weight: bold; color: #444; }
.generate-btn { font-size: 1.3em !important; font-weight: bold !important; min-height: 80px !important; }
.tab-nav { border-bottom: none !important; }
"""

def render_interface() -> gr.Blocks:
    """Render the main Gradio interface."""
    _i18n_key2lang_dict = get_i18n_dict()
    
    with gr.Blocks(title="SoulX-Podcast", theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"), css=CSS) as page:

        # ================= Header =================
        with gr.Row(elem_classes=["header-row", "container"]):
            with gr.Column(scale=8):
                gr.Markdown("# 🎙️ SoulX-Podcast WebUI")
            with gr.Column(scale=4, min_width=200):
                 gr.Markdown("[📖 帮助文档](https://github.com/Y-G-Q/SoulX-Podcast) | [🔗 GitHub](https://github.com/Y-G-Q/SoulX-Podcast)")

        # ================= Main Content =================
        with gr.Row(elem_classes=["container"]):
            
            # ================= LEFT COLUMN: Production Workshop (70%) =================
            with gr.Column(scale=7):
                
                # --- 1. Speaker Settings (Tabs) ---
                # Initialize with default speaker label
                initial_accordion_label = "👥 说话人设置 / Speakers (选择说话人 1)"
                with gr.Accordion(initial_accordion_label, open=False) as speaker_accordion:
                    speakers_state = gr.State(value=1)
                    
                    speaker_checkbox_list = []
                    speaker_remark_list = []
                    speaker_audio_list = []
                    speaker_text_list = []
                    speaker_dialect_list = []
                    speaker_tabs_list = [] # List of Tab components to toggle visibility
                    
                    # 操作区置顶
                    with gr.Row():
                        add_speaker_btn = gr.Button(f"➕ {i18n('add_speaker_btn_label')}", variant="secondary", scale=2)
                        with gr.Group():
                            with gr.Row():
                                quick_add_num = gr.Number(
                                    label="", 
                                    value=1, 
                                    minimum=1, 
                                    maximum=MAX_SPEAKERS,
                                    step=1,
                                    precision=0,
                                    scale=1,
                                    container=False,
                                    min_width=60
                                )
                                quick_add_btn = gr.Button(f"🚀 {i18n('quick_add_btn_label')}", variant="primary", scale=2, min_width=80)
                        
                        batch_delete_btn = gr.Button(f"🗑️ {i18n('batch_delete_btn_label')}", variant="stop", scale=1)
                        
                        # Select all/none buttons (small)
                        select_all_btn = gr.Button(f"☑️", variant="secondary", scale=0, min_width=50)
                        select_none_btn = gr.Button(f"☐", variant="secondary", scale=0, min_width=50)
                    
                    speaker_selection_group = gr.CheckboxGroup(
                        label="快速勾选要删除的说话人",
                        choices=[get_speaker_display_label(1)],
                        value=[],
                        interactive=True,
                    )
                    
                    with gr.Tabs() as speaker_tabs_container:
                        for i in range(MAX_SPEAKERS):
                            tab_label = get_speaker_display_label(i + 1)
                            with gr.Tab(label=tab_label, visible=(i < 1)) as tab:
                                group, checkbox, remark, audio, text, dialect = create_speaker_group(i + 1)
                                speaker_checkbox_list.append(checkbox)
                                speaker_remark_list.append(remark)
                                speaker_audio_list.append(audio)
                                speaker_text_list.append(text)
                                speaker_dialect_list.append(dialect)
                                speaker_tabs_list.append(tab)
                                
                                # 备注变化时同步更新 Tab 与复选框标签，以及 Accordion 标题
                                idx_state = gr.State(i + 1)
                                remark.change(
                                    fn=update_single_speaker_label,
                                    inputs=[remark, idx_state],
                                    outputs=[checkbox, tab],
                                ).then(
                                    fn=update_speaker_accordion_label,
                                    inputs=[speakers_state] + speaker_remark_list,
                                    outputs=[speaker_accordion],
                                ).then(
                                    fn=update_speaker_selection_choices,
                                    inputs=[speakers_state] + speaker_remark_list,
                                    outputs=[speaker_selection_group],
                                )
                    
                # --- 2. Dialogue Input ---
                gr.Markdown("### 📝 对话内容 / Dialogue", elem_classes=["section-header"])
                
                num_text_inputs_state = gr.State(value=1)
                
                # Number of inputs selector
                with gr.Row():
                     num_text_inputs_selector = gr.Number(
                        label="输入框数量 / Number of Inputs",
                        value=1,
                        minimum=1,
                        maximum=MAX_TEXT_INPUTS,
                        step=1,
                        precision=0,
                        interactive=True,
                        scale=1,
                     )
                     diff_spk_pause_input = gr.Number(
                        label="不同说话人停顿(ms)",
                        value=0,
                        minimum=0,
                        step=50,
                        interactive=True,
                        scale=1,
                     )
                     task_pause_input = gr.Number(
                        label="任务间停顿(ms)",
                        value=500,
                        minimum=0,
                        step=100,
                        interactive=True,
                        scale=1,
                     )

                # Main text input area
                dialogue_text_inputs_list = []
                dialogue_audio_preview_list = [] # Audio previews for each task
                dialogue_download_list = []      # Download files for each task
                
                # We still need the list for the backend logic
                with gr.Group():
                     for i in range(MAX_TEXT_INPUTS):
                        dialogue_text_input = gr.Code(
                            label=f"{i18n('dialogue_text_input_label')} {i+1}",
                            value="",
                            language="javascript",
                            lines=6,
                            max_lines=12,
                            interactive=True,
                            show_line_numbers=True,
                            visible=(i < 1),
                        )
                        dialogue_text_inputs_list.append(dialogue_text_input)
                        
                        # Audio preview component - directly below the text input
                        # The preview will show audio when it's generated, no progress info
                        preview = gr.Audio(
                            label=f"任务 {i+1} 音频预览" if get_language() == "zh" else f"Task {i+1} Audio Preview",
                            visible=(i < 1),  # Same visibility as text input
                            interactive=False,
                            show_download_button=True,
                            value=None  # No audio initially
                        )
                        download = gr.File(
                            label=f"任务 {i+1} 下载" if get_language() == "zh" else f"Task {i+1} Download",
                            visible=False
                        )
                        dialogue_audio_preview_list.append(preview)
                        dialogue_download_list.append(download)

                # Update inputs visibility when number changes
                num_text_inputs_selector.change(
                    fn=update_text_inputs_visibility,
                    inputs=[num_text_inputs_selector],
                    outputs=[num_text_inputs_state] + dialogue_text_inputs_list + dialogue_audio_preview_list + dialogue_download_list
                )

                # --- 3. Generate Button ---
                gr.Markdown("### ⚙️ 生成 / Generate", elem_classes=["section-header"])
                
                generate_btn = gr.Button(
                    value=i18n("generate_btn_label"),
                    variant="primary",
                    elem_classes=["generate-btn"],
                )

            # ================= RIGHT COLUMN: Finished Goods Warehouse (30%) =================
            with gr.Column(scale=3):
                
                # --- Config Management (Collapsed Menu) ---
                with gr.Accordion("🛠️ 配置管理 / Config", open=False):
                    gr.Markdown("**导入配置**")
                    config_file_choices = list_config_files()
                    with gr.Tabs():
                        with gr.Tab("选择预设"):
                            config_dropdown = gr.Dropdown(
                                label="选择配置文件",
                                choices=config_file_choices,
                                value=config_file_choices[0] if config_file_choices else None,
                                interactive=True
                            )
                            with gr.Row():
                                refresh_config_list_btn = gr.Button("刷新", size="sm")
                                load_selected_config_btn = gr.Button("加载", variant="primary", size="sm")
                        with gr.Tab("上传文件"):
                            import_config_uploader = gr.File(label="JSON文件", file_types=[".json"])
                            load_uploaded_config_btn = gr.Button("加载上传", variant="primary", size="sm")
                    load_selected_status = gr.Textbox(label="加载状态", interactive=False, lines=2)
                    load_uploaded_status = gr.Textbox(visible=False) # Hidden status for upload

                    gr.Markdown("---")
                    gr.Markdown("**导出配置**")
                    export_config_name_input = gr.Textbox(
                        label="配置名称（可选）",
                        placeholder="留空将使用默认名称",
                        lines=1,
                        interactive=True
                    )
                    with gr.Row():
                        export_config_btn = gr.Button("导出当前配置", size="sm")
                        export_config_file = gr.File(label="导出文件", interactive=False, height=50)
                    export_config_status = gr.Textbox(label="状态", interactive=False, lines=1, show_label=False)

                # --- Global Settings (Collapsed Menu) ---
                with gr.Accordion("⚙️ 全局设置 / Global Settings", open=False):
                    lang_choice = gr.Dropdown(
                        choices=["中文", "English"],
                        value="中文",
                        label="语言/Language",
                        interactive=True,
                        scale=1
                    )
                    seed_input = gr.Number(
                        label="Seed (种子)",
                        value=1988,
                        step=1,
                        interactive=True,
                        scale=1,
                    )

                # --- Output Area ---
                gr.Markdown("### 🔊 当前结果 / Output", elem_classes=["section-header"])
                
                # Generate button above output area
                generate_btn_right = gr.Button(
                    value=i18n("generate_btn_label"),
                    variant="primary",
                    elem_classes=["generate-btn"],
                )
                
                generate_audio = gr.Audio(
                    label="完整音频",
                    interactive=False,
                    show_download_button=True
                )
                
                # --- History / Details ---
                gr.Markdown("### 📜 历史记录 / History", elem_classes=["section-header"])
                
                # Using the textbox to show details/history log as requested in wireframe logic (list)
                # But since we don't have a real list component backed by data, we keep the textbox info
                # and maybe the download file.
                
                separated_files_info = gr.Textbox(
                    label="生成日志",
                    show_label=False,
                    interactive=False,
                    lines=20,
                    visible=True,
                    elem_id="history-log"
                )
                
                download_file = gr.File(
                    label="下载全部 (ZIP)",
                    visible=False,
                )

        # ================= Event Handlers =================
        
        # Speaker Management
        # Note: We pass speaker_tabs_list instead of speaker_columns to toggle visibility of Tabs
        add_speaker_btn.click(
            fn=add_speaker,
            inputs=[speakers_state] + speaker_remark_list,
            outputs=[speakers_state] + speaker_checkbox_list + speaker_tabs_list
        ).then(
            fn=update_speaker_accordion_label,
            inputs=[speakers_state] + speaker_remark_list,
            outputs=[speaker_accordion],
        ).then(
            fn=update_speaker_selection_choices,
            inputs=[speakers_state] + speaker_remark_list,
            outputs=[speaker_selection_group],
        )
        
        # Keep quick add logic compatible
        quick_add_btn.click(
            fn=quick_add_speakers,
            inputs=[speakers_state, quick_add_num] + speaker_remark_list,
            outputs=[speakers_state] + speaker_checkbox_list + speaker_tabs_list
        ).then(
            fn=update_speaker_accordion_label,
            inputs=[speakers_state] + speaker_remark_list,
            outputs=[speaker_accordion],
        ).then(
            fn=update_speaker_selection_choices,
            inputs=[speakers_state] + speaker_remark_list,
            outputs=[speaker_selection_group],
        )
        
        select_all_btn.click(
            fn=select_all_checkboxes,
            inputs=[speakers_state],
            outputs=speaker_checkbox_list
        ).then(
            fn=select_all_selection_group,
            inputs=[speakers_state] + speaker_remark_list,
            outputs=[speaker_selection_group],
        )
        
        select_none_btn.click(
            fn=select_none_checkboxes,
            inputs=[speakers_state],
            outputs=speaker_checkbox_list
        ).then(
            fn=select_none_selection_group,
            outputs=[speaker_selection_group],
        )
        
        # Batch Delete
        all_speaker_inputs_for_delete = []
        for i in range(MAX_SPEAKERS):
            all_speaker_inputs_for_delete.extend([
                speaker_checkbox_list[i],
                speaker_audio_list[i],
                speaker_text_list[i],
                speaker_dialect_list[i],
                speaker_remark_list[i],
            ])
        
        all_speaker_outputs_for_delete = []
        for i in range(MAX_SPEAKERS):
            all_speaker_outputs_for_delete.extend([
                speaker_checkbox_list[i],
                speaker_audio_list[i],
                speaker_text_list[i],
                speaker_dialect_list[i],
                speaker_remark_list[i],
            ])
        
        batch_delete_btn.click(
            fn=batch_delete_speakers,
            inputs=[speakers_state] + all_speaker_inputs_for_delete,
            outputs=[speakers_state] + all_speaker_outputs_for_delete + speaker_tabs_list
        ).then(
            fn=update_speaker_accordion_label,
            inputs=[speakers_state] + speaker_remark_list,
            outputs=[speaker_accordion],
        ).then(
            fn=update_speaker_selection_choices,
            inputs=[speakers_state] + speaker_remark_list,
            outputs=[speaker_selection_group],
        )

        # 快捷勾选同步
        speaker_selection_group.change(
            fn=selection_group_to_checkboxes,
            inputs=[speaker_selection_group, speakers_state] + speaker_remark_list,
            outputs=speaker_checkbox_list,
        )

        # Config Events
        all_speaker_inputs_for_config = []
        for i in range(MAX_SPEAKERS):
            all_speaker_inputs_for_config.extend([
                speaker_audio_list[i],
                speaker_text_list[i],
                speaker_dialect_list[i],
                speaker_remark_list[i],
            ])

        export_config_btn.click(
            fn=export_current_config,
            inputs=[
                lang_choice,
                seed_input,
                diff_spk_pause_input,
                task_pause_input,
                speakers_state,
                num_text_inputs_state,
                export_config_name_input,
                *dialogue_text_inputs_list,
                *all_speaker_inputs_for_config,
            ],
            outputs=[export_config_file, export_config_status],
        )

        refresh_config_list_btn.click(
            fn=refresh_config_dropdown,
            inputs=[config_dropdown],
            outputs=[config_dropdown],
        )

        load_uploaded_config_btn.click(
            fn=load_uploaded_and_apply,
            inputs=[import_config_uploader],
            outputs=[
                speakers_state,
                num_text_inputs_state,
                num_text_inputs_selector,
                seed_input,
                diff_spk_pause_input,
                task_pause_input,
                *speaker_checkbox_list,
                *speaker_audio_list,
                *speaker_text_list,
                *speaker_dialect_list,
                *speaker_remark_list,
                *speaker_tabs_list, # Updated to tabs
                *dialogue_text_inputs_list,
                load_uploaded_status,
            ],
        ).then(
            fn=refresh_all_speaker_labels_after_load,
            inputs=[speakers_state] + speaker_remark_list,
            outputs=[*speaker_checkbox_list, *speaker_tabs_list],
        ).then(
            fn=update_speaker_accordion_label,
            inputs=[speakers_state] + speaker_remark_list,
            outputs=[speaker_accordion],
        ).then(
            fn=update_speaker_selection_choices,
            inputs=[speakers_state] + speaker_remark_list,
            outputs=[speaker_selection_group],
        )

        load_selected_config_btn.click(
            fn=load_selected_and_apply,
            inputs=[config_dropdown],
            outputs=[
                speakers_state,
                num_text_inputs_state,
                num_text_inputs_selector,
                seed_input,
                diff_spk_pause_input,
                task_pause_input,
                *speaker_checkbox_list,
                *speaker_audio_list,
                *speaker_text_list,
                *speaker_dialect_list,
                *speaker_remark_list,
                *speaker_tabs_list, # Updated to tabs
                *dialogue_text_inputs_list,
                load_selected_status,
            ],
        ).then(
            fn=refresh_all_speaker_labels_after_load,
            inputs=[speakers_state] + speaker_remark_list,
            outputs=[*speaker_checkbox_list, *speaker_tabs_list],
        ).then(
            fn=update_speaker_accordion_label,
            inputs=[speakers_state] + speaker_remark_list,
            outputs=[speaker_accordion],
        ).then(
            fn=update_speaker_selection_choices,
            inputs=[speakers_state] + speaker_remark_list,
            outputs=[speaker_selection_group],
        )

        # Generate Events
        all_speaker_inputs = []
        for i in range(MAX_SPEAKERS):
            all_speaker_inputs.extend([
                speaker_audio_list[i],
                speaker_text_list[i],
                speaker_dialect_list[i],
                speaker_remark_list[i]  # 添加备注信息
            ])
        
        # Left column generate button
        generate_btn.click(
            fn=collect_and_synthesize_queue,
            inputs=(
                [num_text_inputs_state] +
                [speakers_state, seed_input, diff_spk_pause_input, task_pause_input, lang_choice] +
                dialogue_text_inputs_list +
                all_speaker_inputs
            ),
            outputs=[
                generate_audio,
                separated_files_info,
                download_file,
                generate_btn,
                generate_btn_right,
                *dialogue_audio_preview_list,
                *dialogue_download_list,
            ],
        )
        
        # Right column generate button (same function)
        generate_btn_right.click(
            fn=collect_and_synthesize_queue,
            inputs=(
                [num_text_inputs_state] +
                [speakers_state, seed_input, diff_spk_pause_input, task_pause_input, lang_choice] +
                dialogue_text_inputs_list +
                all_speaker_inputs
            ),
            outputs=[
                generate_audio,
                separated_files_info,
                download_file,
                generate_btn,
                generate_btn_right,
                *dialogue_audio_preview_list,
                *dialogue_download_list,
            ],
        )
        
        # Language Switch
        # Note: We need to update this to handle the new component structure if necessary
        # The change_component_language function returns a long list of updates.
        # We need to make sure the inputs/outputs match exactly what that function expects.
        # Since I changed some components (like Tabs instead of Columns), I should check if
        # change_component_language updates visibility of columns.
        
        # Checking callbacks.py: change_component_language returns updates for labels mainly.
        # It does NOT seem to return updates for the speaker columns/tabs visibility directly, 
        # but it returns updates for labels of inputs.
        # Let's verify the list length.
        
        # The function returns:
        # checkbox_updates (MAX_SPEAKERS)
        # input_updates (MAX_SPEAKERS * 3)
        # dialogue inputs (MAX_TEXT_INPUTS)
        # dialogue previews/downloads (MAX_TEXT_INPUTS * 2)
        # fixed updates (11 items)
        
        # The outputs list in the original code was:
        # speaker_checkbox_list + all_speaker_inputs + dialogue_text_inputs_list + 
        # dialogue_audio_preview_list + dialogue_download_list + [fixed_list]
        
        # This structure seems preserved in my variables.
        # speaker_checkbox_list is same.
        # all_speaker_inputs is same.
        # dialogue_text_inputs_list is same.
        # ...
        # So it should work fine, as it doesn't touch the Tabs/Columns themselves.
        
        lang_choice.change(
            fn=change_component_language,
            inputs=[lang_choice] + speaker_remark_list,
            outputs=(
                speaker_checkbox_list +
                all_speaker_inputs +
                dialogue_text_inputs_list +
                dialogue_audio_preview_list +
                dialogue_download_list +
                [
                    generate_btn,
                    generate_btn_right,  # Right column generate button
                    generate_audio,
                    add_speaker_btn,
                    quick_add_num, # hidden but exists
                    quick_add_btn, # hidden but exists
                    select_all_btn,
                    select_none_btn,
                    batch_delete_btn,
                    separated_files_info,
                    download_file,
                    diff_spk_pause_input,
                    task_pause_input,
                ]
            ),
        )
    
    return page
