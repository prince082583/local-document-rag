"""Generate sample company policy documents (PDF and TXT) for testing the RAG pipeline."""
import os

POLICY_TEXT = """ACME Corporation - Global Security & Authentication Policy (v2.4)

1. Purpose and Scope
This document outlines the mandatory authentication, access control, and identity management rules for all employees, contractors, and automated services accessing ACME corporate assets and databases.

2. Authentication Policy
- Multi-Factor Authentication (MFA): All user and administrative access to internal services, email, cloud infrastructure, and databases requires mandatory Multi-Factor Authentication (MFA) using FIDO2 hardware keys or approved mobile authenticator applications. SMS-based 2FA is explicitly forbidden due to SIM-swapping vulnerabilities.
- Password Standards: Passwords must be at least 16 characters in length and comprise uppercase letters, lowercase letters, numbers, and symbols. Passwords must not be reused across corporate and personal accounts.
- Password Expiration: Master credentials must be rotated every 90 days. Reusing any of the last 10 passwords is blocked by directory services.
- Account Lockout: System accounts are locked out automatically after 5 consecutive failed authentication attempts. A security alert is generated to the Security Operations Center (SOC).
- Session Duration and Inactivity: Active web sessions and CLI credentials automatically expire after 15 minutes of inactivity, requiring re-authentication.

3. Remote Access & Network Security
- All remote access must originate from corporate-managed endpoints and pass through the corporate Zero-Trust Network Access (ZTNA) gateway.
- Public Wi-Fi access requires active VPN tunneling at all times.
- Split-tunneling is disabled on all corporate devices.

4. Incident Reporting & Violations
Any suspected credential compromise or unauthorized access attempt must be reported immediately to security@acmecorp.internal within 1 hour of discovery.
"""

def generate_txt():
    with open("company_policy.txt", "w", encoding="utf-8") as f:
        f.write(POLICY_TEXT)
    print("Created company_policy.txt")

def generate_pdf():
    # Construct a valid, well-formed PDF 1.4 file with standard Helvetica font
    lines = POLICY_TEXT.strip().split("\n")
    # Escape parentheses and backslashes for PDF string literals
    pdf_lines = []
    y = 750
    for line in lines:
        clean = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if clean.strip():
            pdf_lines.append(f"1 0 0 1 50 {y} Tm ({clean}) Tj")
        y -= 14
        if y < 50:
            break

    stream_content = "BT\n/F1 10 Tf\n" + "\n".join(pdf_lines) + "\nET\n"
    stream_bytes = stream_content.encode("latin-1", errors="replace")
    stream_len = len(stream_bytes)

    objects = []
    # obj 1: Catalog
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # obj 2: Pages
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    # obj 3: Page
    objects.append(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n")
    # obj 4: Font
    objects.append(b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    # obj 5: Contents
    obj5_header = f"5 0 obj\n<< /Length {stream_len} >>\nstream\n".encode("latin-1")
    obj5_footer = b"\nendstream\nendobj\n"
    objects.append(obj5_header + stream_bytes + obj5_footer)

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    body = b""
    offsets = []
    current_offset = len(header)

    for obj in objects:
        offsets.append(current_offset)
        body += obj
        current_offset += len(obj)

    xref_offset = len(header) + len(body)
    xref = f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1")
    for offset in offsets:
        xref += f"{offset:010d} 00000 n \n".encode("latin-1")

    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1")

    with open("company_policy.pdf", "wb") as f:
        f.write(header + body + xref + trailer)
    print("Created company_policy.pdf")

if __name__ == "__main__":
    generate_txt()
    generate_pdf()
