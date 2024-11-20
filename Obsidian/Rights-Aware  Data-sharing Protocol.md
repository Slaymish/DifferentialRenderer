# Rights-Aware Data-sharing Protocol and Client Application

- Cryptographically enforced
- Protocol designed to embed legally binding, user-defined rights into the data itself, ensuring compliance even after the data is shared or downloaded
- **They can't access the things they want to recieve/establish a connection, without agreeing to the same terms**

## Components

1. Consent Management
2. Persistent rights metadata
3. Decentralised verification
4. Peer-to-peer communication
5. Accountability and traceability
6. Legal binding


## Product

1. Website

- Acts as hub for agreement creation and cryptographic proof management
- Provides users with tools to:
	- Define terms and conditions
	- Generate and manage cryptographic keys
	- Initiate connects with other users

2. Client application

- Packages data with embedded rights metadata
- Encrypts and send data using the protocol
- Allows recipients to view and verify consent details
- Only only users to view the data someone send once they have agreed to the same terms and conditions to make it legally binding


3. Blockchain integration

- Records agreements and key transactions, ensuring decentralised verification
- Provides tools for auditing data usage and proving rights violations

## Example

1. Alice uploads image to website and specifies:

- Terms: "Non-commercial use only, no third-party sharing"
- Consent is hashed, signed, and stored on blockchain

2. Alice shares image with Bob via client

- Image includes metadata with consent terms, cryptographic proof, and a hash for integrity verification

3. Bob downloads image and tries to share with third-party

- Third party client verifies the metadata and denies access because the term prohibit third-party sharing

4. If Bob bypasses the system, Alice has cryptographic proof to demonstrate a violation




## Cryptographic Example

- $K_A$,$K_B$ - Alice and bobs public key
- $k_A$,$k_B$ - Alice and bobs private key
- $T()$ - timestamp function
- $H(x)$ - Hashing function ($x$ is message to be hashed) (could be SHA256 etc)
- $MAC(m,k)$ - Message Authentication code, hashing $m$ with shared key $k$
- $SEnc(m,k)$ - Symmetrically Encrypt a message ($m$), with key ($k$)
- $SDec(m,k)$ - Symmetrically Decrypt a message ($m$), with key ($k$)
- $AEnc(m,k)$ - Asymmetrically Encrypt a message ($m$), with key ($k$)
- $ADec(m,k)$ - Asymmetrically Decrypt a message ($m$), with key ($k$)
- $t$ - Terms bob and alice wish to agree to

**Scenario: Alice sending image $m$ to Bob**

- Alice goes to website, creates terms for sending, and signs hash with her private key

$$t_{signed} = AEnc(H(T() || t),k_A)$$

- Alice signed terms to bob

$$k={symmetric\ key}$$

$$m_{enc}=t_{signed}||AEnc(k,K_B)$$



Maybe use DHKE to generate shared key? then use that to signed the terms with (so then both of their signed things will be that same)


## Cryptographic idea

1. **Shared Secret and Key Derivation**:

- Both using their private keys, derive a shared key to use
$$S=(B^amod  p)=(A^bmod  p),k=HKDF(S,'data encryption')$$

2. **Hashing and Signing Terms**:

- Alice agrees to terms, stored signature of hashed timestamp and terms
$$    t_{signed}=AEnc(H(T() ∣∣ t),k_A)$$
3. **Encrypting Data**:

- Using hybrid encryption, encrypt shared symmetric key using Bob public key
- Encrypt *message* with symmetric key
$$m_{enc}=SEnc(m,k),k_{enc}=AEnc(k,K_B)$$
4. **Sending Data**:

- Concatenate the terms signature, then hash

$$data_{send}=t_{signed} ∣∣ k_{enc} ∣∣ m_{enc} || \ H(data_{send})$$
5. **Receiving and Verifying Data**:

- First verify hash/integrity of entire message, then
    
    - $k_{dec}=ADec(k_{enc},k_B)$
    - $m_{dec}=SDec(m_{enc},k_{dec})$
    - $h_{terms_{dec}}=ADec(t_{signed},K_A)$

1. **Verification of Terms**:
    
- $h_{terms_{verification}}=H(T() ∣∣ t)$
- if $h_{terms_{dec}}=h_{terms_{verification}}$, **then consent is valid.**


- This currently doesn't require **two way, mandatory consent, just confirmation on bobs end that alice has consented**

