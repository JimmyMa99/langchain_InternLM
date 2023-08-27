import gradio as gr
import shutil
# import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from chains.local_doc_qa import LocalDocQA
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint
from configs.model_config import *
#*************************一些全局变量*************************************#
local_doc_qa = LocalDocQA()
flag_csv_logger = gr.CSVLogger()
embedding_model_dict_list = list(embedding_model_dict.keys())
llm_model_dict_list = list(llm_model_dict.keys())
#################################CSS类定义#####################################
block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""
##################################################################################

##############################一些展示相关的（字体参数）################################
default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)
#########################################################################################

###########################################一些标题提醒###################################
webui_title = """
# mmEngine 知识库问答
📕[https://mmengine.readthedocs.io/en/latest/notes/contributing.html)
"""
# default_vs = get_vs_list()[0] if len(get_vs_list()) > 1 else "为空"
init_message = f"""你好👋,我是基于Langchain+Internlm的mmEngine文档助手，能够方便你快速获取相关的文档细节，更多详情，请参照 https://mmengine.readthedocs.io/en/latest/ 
"""
#########################################################################################
#****************************************************************************************#

#测试函数（不显示）
def test_fn():
    return None


####
def get_answer(query, vs_path, history, mode, score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
               vector_search_top_k=VECTOR_SEARCH_TOP_K, chunk_conent: bool = True,
               chunk_size=CHUNK_SIZE, streaming: bool = STREAMING):
    if mode == "mmEngine知识库问答" and vs_path is not None and os.path.exists(vs_path) and "index.faiss" in os.listdir(vs_path):
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=query, vs_path=vs_path, chat_history=history, streaming=streaming):
            source = "\n\n"
            source += "".join(
                [f"""<details> <summary>出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}</summary>\n"""
                 f"""{doc.page_content}\n"""
                 f"""</details>"""
                 for i, doc in
                 enumerate(resp["source_documents"])])
            history[-1][-1] += source
            yield history, ""
    logger.info(f"flagging: username={FLAG_USER_NAME},query={query},vs_path={vs_path},mode={mode},history={history}")
    # flag_csv_logger.setup([query, vs_path, chatbot, mode], "flagged")
    flag_csv_logger.flag([query, vs_path, history, mode], username=FLAG_USER_NAME)

######################################处理知识库位置###################################################
def get_vs_path(vs_path='./knowledge_base'):
    os.makedirs(vs_path, exist_ok=True)
    return vs_path
#########################################################################################################

#####################################获取模型(init_model)######################################################
def get_model():
    args = parser.parse_args()
    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    llm_model_ins = shared.loaderLLM()
    try:
        local_doc_qa.init_cfg(llm_model=llm_model_ins)
        answer_result_stream_result = local_doc_qa.llm_model_chain(
            {"prompt": 
             "你好", 
             "history": [], "streaming": False})

        for answer_result in answer_result_stream_result['answer_result_stream']:
            print(answer_result.llm_output)
        reply = """模型已成功加载，可以开始对话"""
        logger.info(reply)
        return reply
    except Exception as e:
        logger.error(e)
        reply = """模型未成功加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        if str(e) == "Unknown platform: darwin":
            logger.info("该报错可能因为您使用的是 macOS 操作系统，需先下载模型至本地后执行 Web UI，具体方法请参考项目 README 中本地部署方法及常见问题："
                        " https://github.com/imClumsyPanda/langchain-ChatGLM")
        else:
            logger.info(reply)
        return reply
#####################################################################################################

###################模式更变######################################################################
def change_mode(mode, history):
    if mode == "mmEngine知识库问答":
        return gr.update(visible=True), gr.update(visible=False), history
        # + [[None, "【注意】：您已进入知识库问答模式，您输入的任何查询都将进行知识库查询，然后会自动整理知识库关联内容进入模型查询！！！"]]
    else:
        return gr.update(visible=False), gr.update(visible=False), history
#####################################################################################################

###################存储向量######################################################################
def get_vector_store(vs_id, files, sentence_size, history, one_conent, one_content_segmentation):
    vs_path = vs_id
    filelist = []
    if local_doc_qa.llm_model_chain and local_doc_qa.embeddings:
        if isinstance(files, list):
            # for file in files:
            #     filename = os.path.split(file)[-1]
            #     shutil.move(file, os.path.join(KB_ROOT_PATH, vs_id, "content", filename))
            #     filelist.append(os.path.join(KB_ROOT_PATH, vs_id, "content", filename))
            vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, vs_path, sentence_size)
        else:
            vs_path, loaded_files = local_doc_qa.one_knowledge_add(vs_path, files, one_conent, one_content_segmentation,
                                                                   sentence_size)
        if len(loaded_files):
            file_status = f"已添加 {'、'.join([os.path.split(i)[-1] for i in loaded_files if i])} 内容至知识库，并已加载知识库，请开始提问"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        vs_path = None
    logger.info(file_status)
    return vs_path, None, history + [[None, file_status]], \
           gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path) if vs_path else [])
########################################################################################

with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
    # 定义变量
    # 获取知识库路径
    vs_path=gr.State(get_vs_path())
    # 获取模型
    model_status = gr.State(get_model())
    #展示标题
    gr.Markdown(webui_title)
    # 对话选项卡
    with gr.Tab("对话"):
        with gr.Row():
            with gr.Column(scale=10):
                # 初始化 Chatbot
                chatbot = gr.Chatbot([[None, init_message], [None, model_status.value]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=750)
                # 提问输入框
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交").style(container=False)
            # with gr.Column(scale=5):
            #     # 选择对话模式
            #     mode = gr.Radio(["与Internlm7b对话", "mmEngine知识库问答"],
            #                     label="请选择使用模式",
            #                     value="mmEngine知识库问答", )
            #     knowledge_set = gr.Accordion("知识库设定", visible=False)
            #     vs_setting = gr.Accordion("配置知识库")
            #     mode.change(fn=change_mode,
            #                 inputs=[mode, chatbot],
            #                 outputs=[vs_setting, knowledge_set, chatbot])
            #     with vs_setting:
            #         # 更新已有知识库选项
            #         vs_refresh = gr.Button("更新已有知识库选项")
            
                one_title = gr.Textbox(label="标题", placeholder="请输入要添加单条段落的标题", lines=1)
                one_conent = gr.Textbox(label="内容", placeholder="请输入要添加单条段落的内容", lines=5)
                one_content_segmentation = gr.Checkbox(value=True, label="禁止内容分句入库",
                                                                interactive=True)
                mode = gr.Radio([ "mmEngine知识库问答"],
                label="请选择使用模式",
                value="mmEngine知识库问答", )       
                vs_id='mmEngine'
                vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
                files_path='./knowledge_base/mmEngine/content'
                files=[os.path.join(files_path, i) for i in os.listdir(files_path)]
                if not os.path.exists('./knowledge_base/mmEngine/vector_store'):
                    vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(os.path.join(KB_ROOT_PATH, vs_id, "content"),
                                                                    vs_path, SENTENCE_SIZE)
                # _, loaded_files = local_doc_qa.one_knowledge_add(vs_path, one_conent,  one_conent,one_content_segmentation,
                #                                                 SENTENCE_SIZE)
                    # get_vector_store(vs_path, files, SENTENCE_SIZE, None, None, None)
                vs_path=gr.State(vs_path)
                flag_csv_logger.setup([query, vs_path, chatbot, mode], "flagged")
                    # get_answer(query, vs_path, chatbot, mode,)
                    # query = gr.Textbox(show_label=False,
                    #                placeholder="请输入提问内容，按回车进行提交").style(container=False)
                
                query.submit(get_answer,
                    [query, vs_path, chatbot, mode],
                    [chatbot, query])
            
            

# 启动demo
(demo
 .queue(concurrency_count=3)
 .launch(server_name='0.0.0.0',
         server_port=7860,
         show_api=False,
         share=False,
         inbrowser=False))