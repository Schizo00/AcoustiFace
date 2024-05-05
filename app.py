import gradio as gr
import os
from demo import Pipeline
import pandas as pd


login_state = False
user_role = 'user'


    


    

def process_audio(audio):
    print(audio)
    pipeline = Pipeline("./models/e1_w_ones_w_weightedMSE_w_customL2_12l_lr00001_male_directed_hausdorff/model.h5", "./models/e1_w_ones_w_weightedMSE_w_customL2_12l_lr00001_male_directed_hausdorff/loss_config.pkl")
    return pipeline.__call__(audio)
    # return audio[1][1].shape


with gr.Blocks(visible=True) as main:
    # Login box
    with gr.Row():
        with gr.Column():
            username = gr.Textbox(label="Username", placeholder="username", type='text', visible=True)
        with gr.Column():
            password = gr.Textbox(label="Password", placeholder="password", type='password', visible=True)
    with gr.Row():
        login_button = gr.Button("Login", visible=True)

    # Login box end


    # Signup box

    with gr.Row():
        with gr.Column():
            username_reg = gr.Textbox(label="Username", placeholder="username", type='text', visible=True)
    with gr.Row():
        with gr.Column():
            password_reg = gr.Textbox(label="Password", placeholder="password", type='password', visible=True)
        with gr.Column():
            password_reg_2 = gr.Textbox(label="Password", placeholder="password", type='password', visible=True)
    with gr.Row():
        signup_button = gr.Button("Signup", visible=True)


    # Signup box end

    # Prediction Screen

    with gr.Row():        
        with gr.Column():
            voice_input_user = gr.Audio(type="filepath",scale=1, visible=False)
            image_input_admin = gr.Image(type='filepath',scale=1, visible=False)
        with gr.Column():
            m3d_output_user = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Model",scale=1, visible=False)
            voice_input_admin = gr.Audio(type="filepath",scale=1, visible=False)

    with gr.Row():
        with gr.Column():
            m3d_conv_output_admin = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Model",scale=1, visible=False)
        with gr.Column():
            m3d_output_admin = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Model",scale=1, visible=False)

    with gr.Row():
        submit_audio_button_user = gr.Button("Submit Audio", visible=False)


    def user_signup(username_reg, password_reg, password_reg_2):
        df = pd.read_csv("./users.csv")
        if username_reg not in df['username'].tolist():
            if password_reg == password_reg_2:
                df2 = {'username': username_reg, 'password': password_reg, 'role': 'user'}
                df = df._append(df2, ignore_index = True)
                df.to_csv("./users.csv")

                gr.Info("Signup Successful!")

                return [
                    gr.Textbox(visible=False),
                    gr.Textbox(visible=False),
                    gr.Textbox(visible=False),
                    gr.Button(visible=False)
                ]
            else:
                raise gr.Error("Passwords Dont Match!")
        else:
            raise gr.Error("Username Already Exists!")


    def user_verification(username, password):
        global user_role
        global login_state

        df = pd.read_csv("./users.csv")
        if username in df['username'].tolist():
            true_password = df.loc[df['username'] == username, 'password'].iloc[0]
            role = df.loc[df['username'] == username, 'role'].iloc[0]
            if true_password == password:
                user_role = role
                login_state = True
            

                if role == "user":
                    return [
                        gr.Textbox(placeholder="username", type='text', visible=False),
                        gr.Textbox(placeholder="password", type='password', visible=False),
                        gr.Audio(type="filepath",scale=1, visible=True), 
                        gr.Image(type='filepath',scale=1, visible=False), 
                        gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Model",scale=1, visible=True), 
                        gr.Audio(type="filepath",scale=1, visible=False),
                        gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Model",scale=1, visible=False),
                        gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Model",scale=1, visible=False),
                        gr.Button("Login", visible=False),
                        gr.Button("Submit Image", visible=False),
                        gr.Button("Submit Audio", visible=False),
                        gr.Button("Submit Audio", visible=True),
                        gr.Textbox(visible=False),
                        gr.Textbox(visible=False),
                        gr.Textbox(visible=False),
                        gr.Button(visible=False)
                    ]
                if role == "admin":
                    return [
                        gr.Textbox(placeholder="username", type='text', visible=False),
                        gr.Textbox(placeholder="password", type='password', visible=False),
                        gr.Audio(type="filepath",scale=1, visible=False), 
                        gr.Image(type='filepath',scale=1, visible=True), 
                        gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Model",scale=1, visible=False), 
                        gr.Audio(type="filepath",scale=1, visible=True),
                        gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Model",scale=1, visible=True),
                        gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Model",scale=1, visible=True),
                        gr.Button("Login", visible=False),
                        gr.Button("Submit Image", visible=True),
                        gr.Button("Submit Audio", visible=True),
                        gr.Button("Submit Audio", visible=False),
                        gr.Textbox(visible=False),
                        gr.Textbox(visible=False),
                        gr.Textbox(visible=False),
                        gr.Button(visible=False)
                    ]
            else:
                raise gr.Error("Invalid Credentials!")
        else:
            raise gr.Error("Invalid Credentials!")


    with gr.Row():
        with gr.Column():
            submit_img_button_admin = gr.Button("Submit Image", visible=False)
            submit_img_button_admin.click(fn=process_audio, inputs=image_input_admin, outputs=m3d_conv_output_admin)
        
        with gr.Column():
            submit_audio_button_admin = gr.Button("Submit Audio", visible=False)
            submit_audio_button_admin.click(fn=process_audio, inputs=voice_input_admin, outputs=m3d_output_admin)

    
    submit_audio_button_user.click(fn=process_audio, inputs=voice_input_user, outputs=m3d_output_user)

    
    login_button.click(fn=user_verification, inputs=[username, password], outputs=[
        username, 
        password, 
        voice_input_user, 
        image_input_admin, 
        m3d_output_user, 
        voice_input_admin,
        m3d_conv_output_admin,
        m3d_output_admin,
        login_button,
        submit_img_button_admin,
        submit_audio_button_admin,
        submit_audio_button_user,
        username_reg,
        password_reg,
        password_reg_2,
        signup_button
        ])
    
    signup_button.click(fn=user_signup, inputs=[username_reg, password_reg, password_reg_2], outputs=[
        username_reg, 
        password_reg,
        password_reg_2,
        signup_button
        ])
    
    




# with gr.Blocks() as demo:
#     with gr.Row():
#         with gr.Column():
#             voice_input = gr.Audio(type="filepath",scale=1)
    
#         with gr.Column():
#             generated_output = gr.Model3D(
#                 clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Model",
#                 scale=1
#             )
    # with gr.Row():
    #     with gr.Column():

    # submit = gr.Button("Submit", scale=0.5)
    # submit.click(fn=process_audio, inputs=voice_input, outputs=generated_output)






# demo = gr.Interface(
#     fn=process_audio,
#     inputs=gr.Audio(type='filepath'),
#     # outputs=gr.Textbox()
#     outputs=gr.Model3D(
#             clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Model"),
#     examples=[
#         [os.path.join(os.path.dirname(__file__), "files/Bunny.obj")],
#         [os.path.join(os.path.dirname(__file__), "files/Duck.glb")],
#         [os.path.join(os.path.dirname(__file__), "files/Fox.gltf")],
#         [os.path.join(os.path.dirname(__file__), "files/face.obj")],
#     ],
# )

if __name__ == "__main__":
    main.launch(share=True)
