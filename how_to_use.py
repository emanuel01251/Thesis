import gradio as gr

def create_how_to_use_page():
    with gr.Blocks(theme='ParityError/Interstellar') as how_to_use:
        gr.Image(value="img/how_to_use.png", show_label=False, container=False, height=150, elem_classes="header-image")

        gr.HTML("""
        <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>How To Use</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        body {
            margin: 0;
            padding: 0;
            background-color: #0a0a0f;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
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

        .section {
            margin: 20px 0;
            padding: 25px;
            background: rgba(20, 20, 24, 0.95);
            border-radius: 10px;
            display: flex;
            align-items: flex-start;
            gap: 24px;
        }

        .icon {
            width: 48px;
            height: 48px;
            flex-shrink: 0;
            stroke: #8e8e9d;
        }

        .content {
            flex-grow: 1;
        }

        .section-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 22px;
            font-weight: 600;
            color: #ffffff;
            margin: 0 0 16px 0;
            letter-spacing: -0.02em;
            position: relative;
            display: inline-block;
        }

        .section-title::after {
            content: '';
            position: absolute;
            bottom: -4px;
            left: 0;
            width: 30px;
            height: 2px;
            background: linear-gradient(90deg, #702cdc, #2cdca7);
            border-radius: 2px;
        }
        li {
            list-style-type: none;
        }
        .steps {
            list-style: none; 
            padding: 0;
            margin: 0;
        }

        .step {
            margin: 12px 0;
            padding-left: 24px;
            position: relative;
            line-height: 1.7;
            color: #c8c8d0;
            font-size: 15px;
            letter-spacing: 0.01em;
        }

        .step::before {
            content: '';
            position: absolute;
            left: 0;
            top: 11px;
            width: 6px;
            height: 6px;
            background: #2cdca7;
            border-radius: 50%;
        }

        .highlight {
            font-family: 'Space Grotesk', sans-serif;
            background: linear-gradient(90deg, #702cdc, #2cdca7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 500;
            letter-spacing: -0.01em;
        }

        .back-button {
            display: inline-flex;
            align-items: center;
            padding: 12px 24px;
            background: linear-gradient(90deg, #702cdc, #2cdca7);
            border-radius: 8px;
            margin-top: 20px;
            font-weight: 500;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: 0.02em;
            font-size: 15px;
            text-decoration: none !important;
        }

        .back-button, .back-button:hover, .back-button:visited, .back-button:active {
            color: white !important;
        }

        .back-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(44, 220, 167, 0.2);
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

        /* Added smooth transitions */
        .section {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .section:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(44, 220, 167, 0.1);
        }

        /* Enhanced focus states for accessibility */
        .back-button:focus {
            outline: none;
            box-shadow: 0 0 0 3px rgba(44, 220, 167, 0.3);
        }

        .header-image {
        margin-top: 30px !important;
        }
                
        @media (max-width: 768px) {
            .container {
                margin: 20px;
                padding: 20px;
            }
            
            .section {
                padding: 20px;
            }

            .section-title {
                font-size: 20px;
            }

            .step {
                font-size: 14px;
            }
        }
    </style>
</head>
<body>
    <div class="background-pattern"></div>
    <div class="container">
        <!-- Getting Started Section -->
        <div class="section">
            <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="16" x2="12" y2="16.01"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
            </svg>
            <div class="content">
                <h2 class="section-title">Getting Started</h2>
                <ul class="steps">
                    <li class="step">Enter your Tagalog-Ilonggo text in the input provided on the main page</li>
                    <li class="step">Select one of four advance BERT-based POS models based on your needs</li>
                    <li class="step">Click the submit button to process your text</li>
                </ul>
            </div>
        </div>

        <!-- Understanding the Models Section -->
        <div class="section">
            <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
            <div class="content">
                <h2 class="section-title">Understanding the Models</h2>
                <ul class="steps">
                    <li class="step"><span class="highlight">SSP with Augmentation:</span> Best for general-purpose tagging with enhanced accuracy through data augmentation</li>
                    <li class="step"><span class="highlight">SOP with Augmentation:</span> Optimized for specific language patterns with augmented training data</li>
                    <li class="step"><span class="highlight">SSP without Augmentation:</span> Base model for standard POS tagging tasks</li>
                    <li class="step"><span class="highlight">SOP without Augmentation:</span> Base model with specialized language pattern recognition</li>
                </ul>
            </div>
        </div>

        <!-- Reading the Results Section -->
        <div class="section">
            <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="18" height="18" rx="2"/>
                <line x1="7" y1="8" x2="17" y2="8"/>
                <line x1="7" y1="12" x2="17" y2="12"/>
                <line x1="7" y1="16" x2="13" y2="16"/>
            </svg>
            <div class="content">
                <h2 class="section-title">Reading the Results</h2>
                <ul class="steps">
                    <li class="step">View the color-coded tags for each word in your text</li>
                    <li class="step">Check the confidence scores to understand the model's certainty</li>
                    <li class="step">You can refer to the POS tag reference tab to view the meaning of each tagsets</li>
                </ul>
            </div>
        </div>

        <a href="/" class="back-button">← Get Started </a>
    </div>
</body>
</html>
        """)
    
    return how_to_use

if __name__ == "__main__":
    page = create_how_to_use_page()
    page.launch()