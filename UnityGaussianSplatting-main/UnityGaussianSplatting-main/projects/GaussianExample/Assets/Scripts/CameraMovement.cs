using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraMovement : MonoBehaviour
{
    public float moveSpeed = 10f; // Speed of movement
    public float lookSpeed = 2f;  // Sensitivity of the mouse look

    private float pitch = 0f;
    private float yaw = 0f;

    void Start()
    {
        // Initialize pitch and yaw based on the current rotation
        Vector3 eulerAngles = transform.rotation.eulerAngles;
        pitch = eulerAngles.x;
        yaw = eulerAngles.y;
    }
    
    void Update()
    {
        // Movement (WASD)
        float moveX = Input.GetAxis("Horizontal"); // A/D or Left/Right Arrow
        float moveZ = Input.GetAxis("Vertical");   // W/S or Up/Down Arrow
        //float moveY = 0;
//
        //if (Input.GetKey(KeyCode.E)) moveY = 1; // E for upward movement
        //if (Input.GetKey(KeyCode.Q)) moveY = -1; // Q for downward movement

        Vector3 move = new Vector3(0, 0, moveZ) * moveSpeed * Time.deltaTime;
        transform.Translate(move);

        // Mouse Look
        if (Input.GetMouseButton(1)) // Right Mouse Button held down
        {
            yaw += Input.GetAxis("Mouse X") * lookSpeed;
            pitch -= Input.GetAxis("Mouse Y") * lookSpeed;

            pitch = Mathf.Clamp(pitch, -90f, 90f); // Limit vertical rotation

            transform.rotation = Quaternion.Euler(pitch, yaw, 0f);
        }
    }
}
