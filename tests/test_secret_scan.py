from scripts.check_no_secrets import scan_text


def test_secret_scan_reports_location_without_value():
    secret = "sk-" + "x" * 30
    findings = scan_text(f"API_KEY={secret}\n", location="private.env")
    assert findings
    assert all(secret not in finding for finding in findings)


def test_secret_scan_allows_empty_and_example_values():
    assert scan_text("API_KEY=\nSERPER_KEY_ID=your-serper-key\n", location=".env.example") == []
