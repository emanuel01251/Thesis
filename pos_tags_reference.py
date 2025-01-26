import gradio as gr

def create_pos_tags_reference():
    with gr.Blocks(theme='ParityError/Interstellar') as pos_tags_reference:
        gr.Image(value="img/pos_list.png", show_label=False, container=False, height=150, elem_classes="header-image")

        gr.HTML("""
                <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>POS Tags Reference</title>
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
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        .pos-table-container {
            max-width: 1000px;
            margin: 20px auto;
            padding: 30px;
            background: rgba(28, 28, 33, 0.95);
            border-radius: 16px;
            box-shadow: 0 25px 50px rgba(0,0,0,0.4);
        }

        .pos-table-title {
            font-family: 'Space Grotesk', sans-serif;
            color: #fff;
            font-size: 24px;
            font-weight: 600;
            text-align: center;
            margin-bottom: 25px;
            padding-bottom: 15px;
            letter-spacing: -0.02em;
            position: relative;
        }

        .pos-table-title::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 60px;
            height: 2px;
            background: linear-gradient(90deg, #702cdc, #2cdca7);
            border-radius: 2px;
        }

        .pos-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 8px;
            margin-top: 10px;
        }

        .pos-table thead tr {
            background: linear-gradient(90deg, #702cdc, #2cdca7);
        }

        .pos-table th {
            color: white;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            text-align: left;
            padding: 16px 20px;
            font-size: 15px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .pos-table tbody tr {
            background: rgba(20, 20, 24, 0.95);
            transition: all 0.3s ease;
        }

        .pos-table tbody tr:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(44, 220, 167, 0.1);
            background: rgba(26, 26, 30, 0.95);
        }

        .pos-table td {
            padding: 16px 20px;
            color: #c8c8d0;
            font-size: 15px;
            line-height: 1.6;
            border-top: 4px solid #0a0a0f;
        }

        .pos-table td:first-child {
            border-radius: 10px 0 0 10px;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 500;
            color: #ffffff;
        }

        .pos-table td:last-child {
            border-radius: 0 10px 10px 0;
        }

        .tag-label {
            font-family: 'Space Grotesk', sans-serif;
            display: inline-block;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 500;
            background: linear-gradient(90deg, rgba(112, 44, 220, 0.2), rgba(44, 220, 167, 0.2));
            color: #fff;
            letter-spacing: 0.02em;
        }

        .example {
            color: #2cdca7;
            font-style: italic;
            font-size: 14px;
        }

        .language-label {
            font-family: 'Space Grotesk', sans-serif;
            color: #702cdc;
            font-weight: 500;
            font-size: 14px;
            display: inline-block;
            margin-top: 4px;
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

        @media (max-width: 768px) {
            .pos-table-container {
                margin: 10px;
                padding: 15px;
            }

            .pos-table th, 
            .pos-table td {
                padding: 12px 10px;
                font-size: 14px;
            }

            .example,
            .language-label {
                font-size: 13px;
            }
        }
                
        .header-image {
            margin-top: 30px !important;
        }
                    
    </style>
</head>
<body>
    <div class="background-pattern"></div>
    <div class="pos-table-container">
        <div class="pos-table-title">Parts of Speech Reference</div>
        <table class="pos-table">
            <thead>
                <tr>
                    <th>Part of Speech</th>
                    <th>Tag</th>
                    <th>Description & Examples</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Verb</td>
                    <td><span class="tag-label">VB</span></td>
                    <td>Action words or state of being<br>
                        <span class="language-label">Tagalog:</span> <span class="example">kumakain, naglalakad, tumatakbo, nagbabasa</span><br>
                        <span class="language-label">Hiligaynon:</span> <span class="example">nagakaon, nagalakat, nagadalagan, nagabasa</span>
                    </td>
                </tr>
                <tr>
                    <td>Noun</td>
                    <td><span class="tag-label">NN</span></td>
                    <td>Names of persons, places, things, or ideas<br>
                        <span class="language-label">Tagalog:</span> <span class="example">bahay, araw, puno, libro, tao, aso</span><br>
                        <span class="language-label">Hiligaynon:</span> <span class="example">balay, adlaw, kahoy, libro, tawo, ido</span>
                    </td>
                </tr>
                <tr>
                    <td>Pronoun</td>
                    <td><span class="tag-label">PRNN</span></td>
                    <td>Words that replace nouns<br>
                        <span class="language-label">Tagalog:</span> <span class="example">ako, ikaw, siya, namin, natin, nila</span><br>
                        <span class="language-label">Hiligaynon:</span> <span class="example">ako, ikaw, siya, namon, naton, nila</span>
                    </td>
                </tr>
                <tr>
                    <td>Determiner</td>
                    <td><span class="tag-label">DET</span></td>
                    <td>Words that modify nouns<br>
                        <span class="language-label">Tagalog:</span> <span class="example">ang, mga, yung, itong, yang</span><br>
                        <span class="language-label">Hiligaynon:</span> <span class="example">ang, mga, ini, ina, sina</span>
                    </td>
                </tr>
                <tr>
                    <td>Adjective</td>
                    <td><span class="tag-label">ADJ</span></td>
                    <td>Words that describe nouns<br>
                        <span class="language-label">Tagalog:</span> <span class="example">maganda, mabait, matangkad, masaya</span><br>
                        <span class="language-label">Hiligaynon:</span> <span class="example">matahum, maayo, mataas, malipayon</span>
                    </td>
                </tr>
                <tr>
                    <td>Adverb</td>
                    <td><span class="tag-label">ADV</span></td>
                    <td>Words that modify verbs, adjectives, or other adverbs<br>
                        <span class="language-label">Tagalog:</span> <span class="example">mabilis, kahapon, ngayon, bukas</span><br>
                        <span class="language-label">Hiligaynon:</span> <span class="example">madasig, kahapon, subong, buas</span>
                    </td>
                </tr>
                <tr>
                    <td>Numerical</td>
                    <td><span class="tag-label">NUM</span></td>
                    <td>Numbers and quantities<br>
                        <span class="language-label">Tagalog:</span> <span class="example">isa, dalawa, tatlo</span><br>
                        <span class="language-label">Hiligaynon:</span> <span class="example">isa, duha, tatlo</span>
                    </td>
                </tr>
                <tr>
                    <td>Conjunction</td>
                    <td><span class="tag-label">CONJ</span></td>
                    <td>Words that connect phrases or clauses<br>
                        <span class="language-label">Tagalog:</span> <span class="example">at, o, pero, ngunit, dahil</span><br>
                        <span class="language-label">Hiligaynon:</span> <span class="example">kag, ukon, pero, tungod, kay</span>
                    </td>
                </tr>
                <tr>
                    <td>Punctuation</td>
                    <td><span class="tag-label">PUNCT</span></td>
                    <td>Punctuation marks<br>
                        <span class="example">. , ? ! ; : " "</span>
                    </td>
                </tr>
                <tr>
                    <td>Foreign Word</td>
                    <td><span class="tag-label">FW</span></td>
                    <td>Words from languages other than Tagalog/Hiligaynon<br>
                        <span class="example">computer, cellphone, internet</span>
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
</body>
</html>
        """)
    
    return pos_tags_reference

if __name__ == "__main__":
    page = create_pos_tags_reference()
    page.launch()