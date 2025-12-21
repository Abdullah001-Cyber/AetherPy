from fpdf import FPDF
from datetime import date

class PDF(FPDF):
    def header(self):
        # Font: Arial bold 12
        self.set_font('Arial', 'B', 12)
        # Name
        self.cell(0, 5, 'Abdullah Adnan', 0, 1, 'L')
        # Font: Arial 10
        self.set_font('Arial', '', 10)
        # Contact Info 
        self.cell(0, 5, 'Lahore, Punjab, Pakistan', 0, 1, 'L')
        self.cell(0, 5, '+92 347 4676020 | LinkedIn Profile', 0, 1, 'L')
        self.cell(0, 5, 'Email: [Insert Your Email Here]', 0, 1, 'L')
        # Line break and horizontal line
        self.ln(5)
        self.line(10, 35, 200, 35)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

# Create PDF object
pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.set_font("Arial", size=11)

# --- Content ---

today = date.today().strftime("%B %d, %Y")

# Recipient Info
pdf.cell(0, 5, today, 0, 1)
pdf.ln(5)
pdf.cell(0, 5, "Hiring Manager", 0, 1)
pdf.cell(0, 5, "Radisson Hotel Group (RHG)", 0, 1)
pdf.cell(0, 5, "Lahore, Pakistan", 0, 1)
pdf.ln(10)

# Salutation
pdf.cell(0, 5, "Re: Application for Information Technology Officer", 0, 1)
pdf.ln(5)
pdf.cell(0, 5, "Dear Hiring Manager,", 0, 1)
pdf.ln(5)

# Body Paragraphs
body_text = (
    "I am writing to express my enthusiastic interest in the Information Technology Officer position at Radisson Hotel Group. "
    "Having reviewed your search for a candidate who combines technical expertise with a \"passion for creating memorable experiences,\" "
    "I am confident that my background in network security, virtualization, and operational management makes me an ideal fit for your dynamic team.\n\n"
    
    "Your job description emphasizes the need for a results-driven approach to complex technical issues. "
    "During my tenure as a Network & Virtualization Intern at NETSOL Technologies Inc., I gained the rigorous troubleshooting experience you are seeking. "
    "I successfully implemented VMware virtualization solutions and configured Microsoft Azure cloud resources, ensuring high availability and system efficiency. "
    "This experience has equipped me with the familiarity with network infrastructure and hardware systems essential to keeping a hotel's operations running smoothly.\n\n"
    
    "Beyond standard IT support, I bring a unique \"security-first\" mindset to the role. As a Certified Ethical Hacker (CEH), "
    "I possess a deep understanding of IT security principles and best practices—a specific requirement listed in your posting. "
    "In an industry where guest data privacy is paramount, my ability to not only troubleshoot systems but also proactively identify vulnerabilities "
    "ensures that RHG's technology initiatives remain both innovative and secure.\n\n"
    
    "However, I understand that an IT Officer at Radisson must also be a collaborative partner. My experience as Chief Revenue Officer at Allied Schools "
    "honed my organizational and communication skills. In this role, I managed confidential records and provided data analysis for strategic decision-making. "
    "This required a high degree of adaptability and the ability to translate technical data into actionable insights for non-technical team members, "
    "mirroring the customer-centric interpersonal skills required to support your various departments and guests.\n\n"
    
    "I am eager to bring my fervor for advancing technological solutions to a company that values character and continuous learning. "
    "I welcome the opportunity to discuss how my technical skills in Python and network maintenance can contribute to the seamless guest experience at Radisson Hotel Group."
)

pdf.multi_cell(0, 6, body_text)

# Sign-off
pdf.ln(10)
pdf.cell(0, 5, "Sincerely,", 0, 1)
pdf.ln(10)
pdf.cell(0, 5, "Abdullah Adnan", 0, 1)

# Output
pdf.output("Abdullah_Adnan_Cover_Letter.pdf")
print("PDF generated successfully: Abdullah_Adnan_Cover_Letter.pdf")