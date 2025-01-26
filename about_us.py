import gradio as gr

def create_about_us_page():
   with gr.Blocks(theme='ParityError/Interstellar') as about_us:
       gr.Image(value="img/about_us.png", show_label=False, container=False, height=100, elem_classes="header-image")

       gr.HTML("""
<!DOCTYPE html>
<html lang="en">
<head>
   <meta charset="UTF-8">
   <meta name="viewport" content="width=device-width, initial-scale=1.0">
   <title>About TAGALONGGO</title>
   <link rel="preconnect" href="https://fonts.googleapis.com">
   <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
   <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
   <style>
       body {
           margin: 0;
           padding: 0;
           background-color: #0a0a0f;
           font-family: 'Outfit', system-ui, -apple-system, sans-serif;
           color: #ffffff;
           min-height: 100vh;
           line-height: 1.6;
           -webkit-font-smoothing: antialiased;
           -moz-osx-font-smoothing: grayscale;
       }

       .container {
           max-width: 1000px;
           margin: 40px auto;
           padding: 50px;
           background: linear-gradient(165deg, rgba(28, 28, 33, 0.92), rgba(17, 24, 39, 0.95));
           backdrop-filter: blur(12px);
           border: 1px solid rgba(108, 44, 220, 0.2);
           border-radius: 24px;
           box-shadow: 
               0 25px 50px rgba(0, 0, 0, 0.3),
               inset 0 2px 6px rgba(255, 255, 255, 0.05);
       }

       .title {
           font-family: 'Outfit', sans-serif;
           text-align: center;
           font-size: 36px;
           margin-bottom: 30px;
           background: linear-gradient(90deg, #702cdc, #2cdca7, #702cdc);
           background-size: 200% auto;
           -webkit-background-clip: text;
           -webkit-text-fill-color: transparent;
           letter-spacing: -0.02em;
           font-weight: 700;
           position: relative;
           text-shadow: 0 2px 10px rgba(108, 44, 220, 0.3);
           animation: gradient 8s linear infinite;
       }

       @keyframes gradient {
           0% { background-position: 0% center; }
           100% { background-position: 200% center; }
       }

       .title::after {
           content: '';
           position: absolute;
           bottom: -10px;
           left: 50%;
           transform: translateX(-50%);
           width: 60px;
           height: 2px;
           background: linear-gradient(90deg, #702cdc, #2cdca7);
           border-radius: 2px;
       }

       .team-section {
           display: grid;
           grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
           gap: 25px;
           margin: 40px 0;
       }

       .team-member {
           background: linear-gradient(145deg, rgba(20, 20, 24, 0.95), rgba(28, 28, 33, 0.98));
           padding: 35px;
           border-radius: 20px;
           text-align: center;
           transition: all 0.3s ease;
           display: flex;
           flex-direction: column;
           align-items: center;
           border: 1px solid rgba(108, 44, 220, 0.15);
           box-shadow: 
               0 10px 30px rgba(0, 0, 0, 0.2),
               inset 0 2px 6px rgba(255, 255, 255, 0.05);
           backdrop-filter: blur(8px);
       }

       .team-member:hover {
           transform: translateY(-5px);
           background: linear-gradient(165deg, rgba(26, 26, 30, 0.98), rgba(31, 41, 55, 0.95));
           box-shadow: 
               0 15px 35px rgba(44, 220, 167, 0.15),
               inset 0 2px 6px rgba(255, 255, 255, 0.08);
       }

       .member-name {
            font-family: 'Outfit', sans-serif;
            font-size: 32px !important;
            font-weight: 700 !important;
            margin: 20px 0 !important;
            background: linear-gradient(90deg, #2cdca7, #702cdc);
            -webkit-background-clip: text !important;
            background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            text-shadow: 0 2px 8px rgba(44, 220, 167, 0.2);
            letter-spacing: -0.02em;
        }
               
       .member-role {
           font-family: 'Outfit', sans-serif;
           color: #2cdca7;
           font-size: 15px;
           letter-spacing: 1px;
           font-weight: 500;
           text-transform: uppercase;
           margin-bottom: 12px;
       }

       .member-description {
           font-size: 15px;
           line-height: 1.7;
           font-weight: 300;
           color: #e2e2e7;
           letter-spacing: 0.2px;
           margin-top: 10px;
           text-align: left;
       }

       .project-description {
           margin: 50px 0;
           line-height: 1.8;
           padding: 40px;
           background: linear-gradient(145deg, rgba(20, 20, 24, 0.92), rgba(28, 28, 33, 0.95));
           border-radius: 20px;
           border: 1px solid rgba(108, 44, 220, 0.1);
           box-shadow: 
               inset 0 2px 4px rgba(0, 0, 0, 0.1),
               0 10px 30px rgba(0, 0, 0, 0.15);
           font-size: 16px;
           letter-spacing: 0.2px;
       }

       .project-description p {
           color: #e2e2e7;
           font-weight: 300;
       }

       .back-button {
           display: inline-flex;
           align-items: center;
           padding: 12px 24px;
           background: linear-gradient(90deg, #702cdc, #2cdca7);
           border-radius: 12px;
           margin-top: 20px;
           font-weight: 500;
           transition: all 0.3s ease;
           border: none;
           cursor: pointer;
           font-family: 'Outfit', sans-serif;
           letter-spacing: 0.02em;
           font-size: 15px;
           text-decoration: none !important;
           box-shadow: 0 4px 15px rgba(44, 220, 167, 0.2);
       }

       .back-button:hover {
           transform: translateY(-2px);
           background: linear-gradient(90deg, #2cdca7, #702cdc);
           box-shadow: 0 6px 20px rgba(44, 220, 167, 0.3);
       }

       .background-pattern {
           position: fixed;
           top: 0;
           right: 0;
           width: 100%;
           height: 100%;
           opacity: 0.05;
           pointer-events: none;
           background-image: radial-gradient(circle, #ffffff 1px, transparent 1px);
           background-size: 30px 30px;
       }
               
        .member-description, .project-description p {
           text-align: justify !important;
           hyphens: auto !important;
           word-spacing: -0.05em !important;
        }
               
        .header-image {
        margin-top: 35px !important;
        }

       @media (max-width: 768px) {
           .container {
               margin: 20px;
               padding: 30px;
           }

           .title {
               font-size: 28px;
           }

           .team-member {
               padding: 25px;
           }

           .project-description {
               padding: 30px;
           }
       }
   </style>
</head>
<body>
   <div class="background-pattern"></div>
   <div class="container">
       <h1 class="title">ABOUT TAGALONGGO POS TAGGER</h1>
       
       <div class="project-description">
           <p>TAGALONGGO POS Tagger is an innovative natural language processing tool designed specifically for 
           Tagalog-Ilonggo text analysis. Our project aims to advance the field of bilingual text processing 
           by providing accurate and efficient part-of-speech tagging for mixed language contexts.</p>
       </div>
       
       <h2 class="title">OUR TEAM</h2>
       <div class="team-section">
           <div class="team-member">
               <h3 class="member-name">Emanuel Jamero</h3>
               <p class="member-role">Lead Developer</p>
               <p class="member-description">
                   Leads the overall development of the TAGALONGGO POS Tagger platform. Specializes in system architecture, 
                   backend development, and integration of machine learning models into production environments.
               </p>
           </div>
           
           <div class="team-member">
               <h3 class="member-name">John Nicolas Oandasan</h3>
               <p class="member-role">ML Engineer</p>
               <p class="member-description">
                   Focuses on developing and optimizing the BERT-based models for POS tagging. Expertise in training 
                   and fine-tuning neural networks for natural language processing tasks.
               </p>
           </div>
           
           <div class="team-member">
               <h3 class="member-name">Vince Favorito</h3>
               <p class="member-role">Data Scientist</p>
               <p class="member-description">
                   Handles data preprocessing, analysis, and validation. Specializes in creating and maintaining 
                   datasets for training and testing, ensuring high-quality bilingual language processing.
               </p>
           </div>
           
           <div class="team-member">
               <h3 class="member-name">Kyla Marie Alcantara</h3>
               <p class="member-role">Research Lead</p>
               <p class="member-description">
                   Leads the research initiatives and linguistic analysis. Expertise in bilingual language patterns 
                   and computational linguistics, ensuring accurate tagging for mixed language contexts.
               </p>
           </div>
       </div>
       
       <div class="project-description">
           <p>Our team combines expertise in machine learning, natural language processing, and linguistics 
           to create a tool that accurately identifies and tags parts of speech in bilingual text. 
           We've developed and implemented various BERT-based models to handle the unique challenges 
           of processing mixed Tagalog-Ilonggo text.</p>
       </div>
       
       <a href="/" class="back-button">← Back to Main Page</a>
   </div>
</body>
</html>
       """)
   
   return about_us

if __name__ == "__main__":
   page = create_about_us_page()
   page.launch()