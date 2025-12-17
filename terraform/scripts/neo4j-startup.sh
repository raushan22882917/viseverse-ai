#!/bin/bash
# Neo4j installation and configuration script for Ubuntu

set -e

# Update system
apt-get update
apt-get install -y wget curl gnupg software-properties-common

# Install Java 11
apt-get install -y openjdk-11-jdk

# Add Neo4j repository
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | tee /etc/apt/sources.list.d/neo4j.list

# Update package list and install Neo4j
apt-get update
apt-get install -y neo4j=1:5.13.0

# Configure Neo4j
NEO4J_CONF="/etc/neo4j/neo4j.conf"

# Backup original config
cp $NEO4J_CONF $NEO4J_CONF.backup

# Configure Neo4j settings
cat >> $NEO4J_CONF << EOF

# VisaVerse Guardian AI Configuration
server.default_listen_address=0.0.0.0
server.bolt.listen_address=0.0.0.0:7687
server.http.listen_address=0.0.0.0:7474
server.https.listen_address=0.0.0.0:7473

# Memory settings
server.memory.heap.initial_size=1G
server.memory.heap.max_size=1G
server.memory.pagecache.size=512M

# Security settings
dbms.security.auth_enabled=true
dbms.security.procedures.unrestricted=gds.*,apoc.*

# Performance settings
dbms.transaction.timeout=60s
dbms.lock.acquisition.timeout=60s

# Logging
dbms.logs.query.enabled=INFO
dbms.logs.query.threshold=1s
EOF

# Set Neo4j password
neo4j-admin dbms set-initial-password "${neo4j_password}"

# Enable and start Neo4j service
systemctl enable neo4j
systemctl start neo4j

# Wait for Neo4j to start
sleep 30

# Install APOC plugin
NEO4J_HOME="/var/lib/neo4j"
APOC_VERSION="5.13.0"
wget -O $NEO4J_HOME/plugins/apoc-$APOC_VERSION-core.jar \
  https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/$APOC_VERSION/apoc-$APOC_VERSION-core.jar

# Restart Neo4j to load plugins
systemctl restart neo4j

# Wait for Neo4j to restart
sleep 30

# Create initial schema and constraints for VisaVerse
cypher-shell -u neo4j -p "${neo4j_password}" << 'CYPHER'
// Create constraints
CREATE CONSTRAINT country_code_unique IF NOT EXISTS FOR (c:Country) REQUIRE c.code IS UNIQUE;
CREATE CONSTRAINT visa_type_id_unique IF NOT EXISTS FOR (v:VisaType) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT document_type_name_unique IF NOT EXISTS FOR (d:DocumentType) REQUIRE d.name IS UNIQUE;
CREATE CONSTRAINT requirement_id_unique IF NOT EXISTS FOR (r:Requirement) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT field_name_unique IF NOT EXISTS FOR (f:Field) REQUIRE f.name IS UNIQUE;

// Create indexes
CREATE INDEX country_name_index IF NOT EXISTS FOR (c:Country) ON (c.name);
CREATE INDEX visa_type_name_index IF NOT EXISTS FOR (v:VisaType) ON (v.name);
CREATE INDEX requirement_type_index IF NOT EXISTS FOR (r:Requirement) ON (r.type);

// Create sample data for common countries and visa types
MERGE (us:Country {code: 'US', name: 'United States'})
MERGE (ca:Country {code: 'CA', name: 'Canada'})
MERGE (gb:Country {code: 'GB', name: 'United Kingdom'})
MERGE (au:Country {code: 'AU', name: 'Australia'})
MERGE (de:Country {code: 'DE', name: 'Germany'})
MERGE (fr:Country {code: 'FR', name: 'France'})
MERGE (jp:Country {code: 'JP', name: 'Japan'})
MERGE (sg:Country {code: 'SG', name: 'Singapore'})

// Create visa types
MERGE (tourist:VisaType {id: 'tourist', name: 'Tourist/Visitor Visa'})
MERGE (business:VisaType {id: 'business', name: 'Business Visa'})
MERGE (work:VisaType {id: 'work', name: 'Work Visa'})
MERGE (student:VisaType {id: 'student', name: 'Student Visa'})
MERGE (family:VisaType {id: 'family', name: 'Family/Spouse Visa'})
MERGE (transit:VisaType {id: 'transit', name: 'Transit Visa'})

// Create document types
MERGE (passport:DocumentType {name: 'passport', required: true})
MERGE (visa_app:DocumentType {name: 'visa_application', required: true})
MERGE (employment:DocumentType {name: 'employment_letter', required: false})
MERGE (bank:DocumentType {name: 'bank_statement', required: false})
MERGE (education:DocumentType {name: 'educational_certificate', required: false})
MERGE (medical:DocumentType {name: 'medical_certificate', required: false})

// Create relationships for US tourist visa as example
MERGE (us)-[:OFFERS]->(tourist)
MERGE (tourist)-[:REQUIRES]->(passport)
MERGE (tourist)-[:REQUIRES]->(visa_app)
MERGE (tourist)-[:RECOMMENDS]->(bank)

// Create fields for passport
MERGE (passport)-[:MUST_HAVE]->(:Field {name: 'full_name', type: 'string', required: true})
MERGE (passport)-[:MUST_HAVE]->(:Field {name: 'passport_number', type: 'string', required: true})
MERGE (passport)-[:MUST_HAVE]->(:Field {name: 'date_of_birth', type: 'date', required: true})
MERGE (passport)-[:MUST_HAVE]->(:Field {name: 'nationality', type: 'string', required: true})
MERGE (passport)-[:MUST_HAVE]->(:Field {name: 'expiry_date', type: 'date', required: true})

// Create basic requirements
MERGE (valid_passport:Requirement {
  id: 'valid_passport',
  type: 'document_validity',
  description: 'Passport must be valid for at least 6 months',
  priority: 1
})

MERGE (complete_application:Requirement {
  id: 'complete_application',
  type: 'form_completeness',
  description: 'Visa application form must be completely filled',
  priority: 1
})

MERGE (financial_proof:Requirement {
  id: 'financial_proof',
  type: 'financial_evidence',
  description: 'Proof of sufficient funds for the trip',
  priority: 2
})
CYPHER

echo "Neo4j installation and configuration completed successfully"
echo "Neo4j is running on:"
echo "  HTTP: http://$(hostname -I | awk '{print $1}'):7474"
echo "  Bolt: bolt://$(hostname -I | awk '{print $1}'):7687"
echo "  Username: neo4j"
echo "  Password: [configured via startup script]"